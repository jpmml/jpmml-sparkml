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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.ClassificationModel;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.dmg.pmml.FieldName;

public class FeatureSchemaUtil {

	private FeatureSchemaUtil(){
	}

	static
	public FeatureSchema createSchema(PredictionModel<?, ?> predictionModel, Map<String, Transformer> columns){
		FieldName targetField;
		List<String> targetCategories = null;

		if(predictionModel instanceof ClassificationModel){
			ClassificationModel<?, ?> classificationModel = (ClassificationModel<?, ?>)predictionModel;

			StringIndexerModel stringIndexerModel = requireTransform(StringIndexerModel.class, columns.get(classificationModel.getLabelCol()));

			targetField = FieldName.create(stringIndexerModel.getInputCol());
			targetCategories = Arrays.asList(stringIndexerModel.labels());
		} else

		{
			targetField = FieldName.create(predictionModel.getLabelCol());
		}

		List<Feature> features = new ArrayList<>();

		VectorAssembler vectorAssembler = requireTransform(VectorAssembler.class, columns.get(predictionModel.getFeaturesCol()));

		String[] inputColumns = vectorAssembler.getInputCols();
		for(String inputColumn : inputColumns){
			List<? extends Feature> inputColumnFeatures = getFeatures(inputColumn, columns);

			features.addAll(inputColumnFeatures);
		}

		FeatureSchema schema = new FeatureSchema(targetField, targetCategories, features);

		return schema;
	}

	static
	private List<? extends Feature> getFeatures(String inputColumn, Map<String, Transformer> columns){
		Transformer transformer = columns.get(inputColumn);

		if(transformer == null){
			FieldName name = FieldName.create(inputColumn);

			return Collections.singletonList(new ContinuousFeature(name));
		} // End if

		if(transformer instanceof OneHotEncoder){
			OneHotEncoder oneHotEncoder = (OneHotEncoder)transformer;

			StringIndexerModel stringIndexer = requireTransform(StringIndexerModel.class, columns.get(oneHotEncoder.getInputCol()));

			List<String> labels = new ArrayList<>(Arrays.asList(stringIndexer.labels()));

			Boolean dropLast = (Boolean)oneHotEncoder.get(oneHotEncoder.dropLast()).get();
			if(dropLast){
				labels.remove(labels.size() - 1);
			}

			FieldName name = FieldName.create(stringIndexer.getInputCol());

			List<Feature> features = new ArrayList<>();
			for(String label : labels){
				features.add(new CategoricalFeature<>(name, label));
			}

			return features;
		}

		throw new IllegalArgumentException();
	}

	static
	private <T extends Transformer> T requireTransform(Class<T> clazz, Transformer transformer){

		if(clazz.isInstance(transformer)){
			return clazz.cast(transformer);
		}

		throw new IllegalArgumentException();
	}
}