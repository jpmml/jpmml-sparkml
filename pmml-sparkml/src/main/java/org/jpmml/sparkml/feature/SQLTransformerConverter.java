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
import java.util.Objects;

import org.apache.spark.ml.feature.SQLTransformer;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Field;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.SchemaException;
import org.jpmml.converter.TypeUtil;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sparkml.AliasExpression;
import org.jpmml.sparkml.DatasetUtil;
import org.jpmml.sparkml.ExpressionTranslator;
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

		List<?> objects = encodeLogicalPlan(encoder, logicalPlan);
		for(Object object : objects){

			if(object instanceof List){
				List<?> features = (List<?>)object;

				features.stream()
					.map(Feature.class::cast)
					.forEach(result::add);
			} else

			if(object instanceof Field){
				Field<?> field = (Field<?>)object;

				String name = field.requireName();

				Feature feature = encoder.createFeature(field);

				encoder.putOnlyFeature(name, feature);

				result.add(feature);
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		return result;
	}

	@Override
	public void registerFeatures(SparkMLEncoder encoder){
		encodeFeatures(encoder);
	}

	static
	public List<?> encodeLogicalPlan(SparkMLEncoder encoder, LogicalPlan logicalPlan){
		List<Object> result = new ArrayList<>();

		List<LogicalPlan> children = JavaConversions.seqAsJavaList(logicalPlan.children());
		for(LogicalPlan child : children){
			encodeLogicalPlan(encoder, child);
		}

		List<Expression> expressions = JavaConversions.seqAsJavaList(logicalPlan.expressions());
		for(Expression expression : expressions){
			org.dmg.pmml.Expression pmmlExpression = ExpressionTranslator.translate(encoder, expression);

			if(pmmlExpression instanceof FieldRef){
				FieldRef fieldRef = (FieldRef)pmmlExpression;

				if(encoder.hasFeatures(fieldRef.requireField())){
					List<Feature> features = encoder.getFeatures(fieldRef.requireField());

					result.add(features);

					continue;
				}

				Field<?> field = ensureField(encoder, fieldRef.requireField());
				if(field != null){
					result.add(field);

					continue;
				}
			}

			String name;

			if(pmmlExpression instanceof AliasExpression){
				AliasExpression aliasExpression = (AliasExpression)pmmlExpression;

				name = aliasExpression.getName();
			} else

			{
				name = FieldNameUtil.create("sql", String.valueOf(expression));
			}

			DataType dataType = DatasetUtil.translateDataType(expression.dataType());

			OpType opType = TypeUtil.getOpType(dataType);

			pmmlExpression = AliasExpression.unwrap(pmmlExpression);

			Visitor visitor = new AbstractVisitor(){

				@Override
				public VisitorAction visit(FieldRef fieldRef){

					if(encoder.hasFeatures(fieldRef.requireField())){
						Feature feature = encoder.getOnlyFeature(fieldRef.requireField());

						if(!Objects.equals(fieldRef.requireField(), feature.getName())){
							fieldRef.setField(feature.getName());
						}
					} else

					{
						ensureField(encoder, fieldRef.requireField());
					}

					return super.visit(fieldRef);
				}
			};
			visitor.applyTo(pmmlExpression);

			DerivedField derivedField = encoder.createDerivedField(name, opType, dataType, pmmlExpression);

			result.add(derivedField);
		}

		return result;
	}

	static
	private Field<?> ensureField(SparkMLEncoder encoder, String name){

		try {
			return encoder.getField(name);
		} catch(SchemaException pmmlSe){

			try {
				return encoder.createDataField(name, name);
			} catch(SchemaException se){
				return null;
			}
		}
	}
}