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
import java.util.List;

import org.apache.spark.ml.feature.SQLTransformer;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.StringFeature;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sparkml.AliasExpression;
import org.jpmml.sparkml.DatasetUtil;
import org.jpmml.sparkml.ExpressionTranslator;
import org.jpmml.sparkml.ExpressionUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;
import scala.collection.JavaConversions;

public class SQLTransformerConverter extends FeatureConverter<SQLTransformer> {

	public SQLTransformerConverter(SQLTransformer sqlTransformer){
		super(sqlTransformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		SQLTransformer transformer = getTransformer();

		String statement = transformer.getStatement();

		SparkSession sparkSession = SparkSession.builder()
			.getOrCreate();

		StructType schema = encoder.getSchema();

		LogicalPlan logicalPlan = DatasetUtil.createAnalyzedLogicalPlan(sparkSession, schema, statement);

		List<Feature> result = new ArrayList<>();

		List<Expression> expressions = JavaConversions.seqAsJavaList(logicalPlan.expressions());
		for(Expression expression : expressions){
			org.dmg.pmml.Expression pmmlExpression = ExpressionTranslator.translate(encoder, expression);

			FieldName name;

			if(pmmlExpression instanceof AliasExpression){
				AliasExpression aliasExpression = (AliasExpression)pmmlExpression;

				name = FieldName.create(aliasExpression.getName());
			} else

			{
				name = FieldNameUtil.create("sql", ExpressionUtil.format(expression));
			}

			DataType dataType = DatasetUtil.translateDataType(expression.dataType());

			OpType opType = ExpressionUtil.getOpType(dataType);

			Visitor visitor = new AbstractVisitor(){

				@Override
				public VisitorAction visit(FieldRef fieldRef){
					FieldName name = fieldRef.getField();

					encoder.getOnlyFeature(name.getValue());

					return super.visit(fieldRef);
				}
			};
			visitor.applyTo(pmmlExpression);

			DerivedField derivedField = encoder.createDerivedField(name, opType, dataType, AliasExpression.unwrap(pmmlExpression));

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
					throw new IllegalArgumentException("Data type " + dataType + " is not supported");
			}

			encoder.putOnlyFeature(name.getValue(), feature);

			result.add(feature);
		}

		return result;
	}

	@Override
	public void registerFeatures(SparkMLEncoder encoder){
		encodeFeatures(encoder);
	}
}
