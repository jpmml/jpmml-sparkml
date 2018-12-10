/*
 * Copyright (c) 2018 Villu Ruusmann
 *
 * This file is part of JPMML-SparkML
 *
 * JPMML-SparkML is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * JPMML-SparkML is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with JPMML-SparkML.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.jpmml.sparkml.feature;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.apache.spark.ml.feature.SQLTransformer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.analysis.UnresolvedAlias;
import org.apache.spark.sql.catalyst.analysis.UnresolvedStar;
import org.apache.spark.sql.catalyst.expressions.Alias;
import org.apache.spark.sql.catalyst.expressions.AttributeReference;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.parser.ParserInterface;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.internal.SessionState;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.StringFeature;
import org.jpmml.sparkml.ExpressionMapping;
import org.jpmml.sparkml.ExpressionTranslator;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;
import scala.collection.JavaConversions;

public class SQLTransformerConverter extends FeatureConverter<SQLTransformer> {

	public SQLTransformerConverter(SQLTransformer sqlTransformer){
		super(sqlTransformer);
	}

	// @todo this is a sad copy-paste of #org.apache.spark.ml.feature.SQLTransformer
	private LogicalPlan analyzed(StructType schema, String statement, String uid)
	{
		SparkSession sparkSession = SparkSession.builder()
				.getOrCreate();
		Dataset<Row> df = sparkSession.createDataFrame(Collections.emptyList(), schema);
		String tableName = uid + "_" + UUID.randomUUID().toString().substring(0, 6);
		String realStatement = statement.replace("__THIS__", tableName);
		df.createOrReplaceTempView(tableName);
		LogicalPlan retval = sparkSession.sql(realStatement).queryExecution().analyzed();
		sparkSession.catalog().dropTempView(tableName);
		return retval;
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		SQLTransformer transformer = getTransformer();

		String statement = transformer.getStatement();

		LogicalPlan logicalPlan = analyzed(encoder.getSchema(), statement, transformer.uid());

		ExpressionTranslator.DataTypeResolver dataTypeResolver = new ExpressionTranslator.DataTypeResolver(){

			@Override
			public DataType getDataType(String name){
				Feature feature = encoder.getOnlyFeature(name);

				return feature.getDataType();
			}
		};

		Map<String, Feature> schemaFeatureMap = encoder.getSchemaFeatures().stream().collect(Collectors.toMap(f -> f.getName().getValue(), f -> f));

		List<Feature> result = new ArrayList<>();

		List<Expression> expressions = JavaConversions.seqAsJavaList(logicalPlan.expressions());
		for(Expression expression : expressions){
			String name;

			if(expression instanceof Alias){
				Alias alias = (Alias)expression;

				expression = alias.child();

				name = alias.name();
			} else
			if(expression instanceof AttributeReference) {
				AttributeReference reference = (AttributeReference)expression;
				name = reference.name();
				Feature feature = schemaFeatureMap.get(name);
				if(null == feature) {
					throw new IllegalArgumentException(name);
				}
				result.add(feature);
				continue;
			} else
			if(expression instanceof UnresolvedAlias){
				UnresolvedAlias unresolvedAlias = (UnresolvedAlias)expression;

				expression = unresolvedAlias.child();

				// XXX
				name = "(" + (expression.toString()).replace("\'", "") + ")";
			} else

			if(expression instanceof UnresolvedStar){
				UnresolvedStar unresolvedStar = (UnresolvedStar)expression;

				List<Feature> features = encoder.getSchemaFeatures();

				result.addAll(features);

				continue;
			} else

			{
				throw new IllegalArgumentException(String.valueOf(expression));
			}

			ExpressionMapping expressionMapping = ExpressionTranslator.translate(expression, dataTypeResolver);

			OpType opType;

			DataType dataType = expressionMapping.getDataType();
			switch(dataType){
				case STRING:
					opType = OpType.CATEGORICAL;
					break;
				case INTEGER:
				case DOUBLE:
					opType = OpType.CONTINUOUS;
					break;
				case BOOLEAN:
					opType = OpType.CATEGORICAL;
					break;
				default:
					throw new IllegalArgumentException();
			}

			org.dmg.pmml.Expression pmmlExpression = expressionMapping.getTo();

			DerivedField derivedField = encoder.createDerivedField(FieldName.create(name), opType, dataType, pmmlExpression);

			Feature feature;

			switch(dataType){
				case STRING:
					feature = new StringFeature(encoder, derivedField);
					break;
				case INTEGER:
				case DOUBLE:
					feature = new ContinuousFeature(encoder, derivedField);
					break;
				case BOOLEAN:
					feature = new BooleanFeature(encoder, derivedField);
					break;
				default:
					throw new IllegalArgumentException();
			}

			encoder.putOnlyFeature(name, feature);

			result.add(feature);
		}

		return result;
	}

	@Override
	public void registerFeatures(SparkMLEncoder encoder){
		encodeFeatures(encoder);
	}
}
