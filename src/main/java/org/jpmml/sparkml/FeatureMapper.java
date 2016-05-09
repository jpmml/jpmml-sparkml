/*
 * Copyright (c) 2016 Villu Ruusmann
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
package org.jpmml.sparkml;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.ClassificationModel;
import org.apache.spark.ml.param.shared.HasOutputCol;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.FieldName;

public class FeatureMapper {

	private StructType schema = null;

	private Map<String, List<Feature>> columnFeatures = new LinkedHashMap<>();


	public FeatureMapper(StructType schema){
		setSchema(schema);
	}

	public void append(FeatureConverter<?> converter){
		Transformer transformer = converter.getTransformer();

		List<Feature> features = converter.encodeFeatures(this);

		if(transformer instanceof HasOutputCol){
			HasOutputCol hasOutputCol = (HasOutputCol)transformer;

			String outputCol = hasOutputCol.getOutputCol();

			this.columnFeatures.put(outputCol, features);
		}
	}

	public FeatureSchema createSchema(PredictionModel<?, ?> predictionModel){
		FieldName targetField;
		List<String> targetCategories = null;

		if(predictionModel instanceof ClassificationModel){
			CategoricalFeature targetFeature = (CategoricalFeature)getOnlyFeature(predictionModel.getLabelCol());

			targetField = targetFeature.getName();
			targetCategories = (List)targetFeature.getValue();
		} else

		{
			Feature targetFeature = getOnlyFeature(predictionModel.getLabelCol());

			targetField = targetFeature.getName();
		}

		List<Feature> features = getFeatures(predictionModel.getFeaturesCol());

		FeatureSchema result = new FeatureSchema(targetField, targetCategories, features);

		return result;
	}

	public Feature getOnlyFeature(String column){
		List<Feature> features = getFeatures(column);

		return Iterables.getOnlyElement(features);
	}

	public List<Feature> getFeatures(String column){
		List<Feature> features = this.columnFeatures.get(column);

		if(features == null){
			StructField field = this.schema.apply(column);

			FieldName name = FieldName.create(column);

			Feature feature;

			DataType dataType = field.dataType();
			if((DataTypes.IntegerType).sameType(dataType) || (DataTypes.FloatType).sameType(dataType) || (DataTypes.DoubleType).sameType(dataType)){
				feature = new ContinuousFeature(name);
			} else

			{
				feature = new PseudoFeature(name);
			}

			return Collections.singletonList(feature);
		}

		return features;
	}

	public StructType getSchema(){
		return this.schema;
	}

	private void setSchema(StructType schema){
		this.schema = schema;
	}
}