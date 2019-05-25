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
import java.util.List;

import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.shared.HasFeaturesCol;
import org.apache.spark.ml.param.shared.HasLabelCol;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.apache.spark.ml.param.shared.HasProbabilityCol;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.LabelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;

abstract
public class ClassificationModelConverter<T extends PredictionModel<Vector, T> & HasLabelCol & HasFeaturesCol & HasPredictionCol> extends ModelConverter<T> {

	public ClassificationModelConverter(T model){
		super(model);
	}

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.CLASSIFICATION;
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, SparkMLEncoder encoder){
		T model = getTransformer();

		CategoricalLabel categoricalLabel = (CategoricalLabel)label;

		List<Integer> categories = LabelUtil.createTargetCategories(categoricalLabel.size());

		List<OutputField> result = new ArrayList<>();

		String predictionCol = model.getPredictionCol();

		OutputField pmmlPredictedField = ModelUtil.createPredictedField(FieldName.create("pmml(" + predictionCol + ")"), categoricalLabel.getDataType(), OpType.CATEGORICAL)
			.setFinalResult(false);

		result.add(pmmlPredictedField);

		MapValues mapValues = PMMLUtil.createMapValues(pmmlPredictedField.getName(), categoricalLabel.getValues(), categories)
			.setDataType(DataType.DOUBLE);

		OutputField predictedField = new OutputField(FieldName.create(predictionCol), DataType.DOUBLE)
			.setOpType(OpType.CATEGORICAL)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(mapValues);

		result.add(predictedField);

		encoder.putOnlyFeature(predictionCol, new IndexFeature(encoder, predictedField, categories));

		if(model instanceof HasProbabilityCol){
			HasProbabilityCol hasProbabilityCol = (HasProbabilityCol)model;

			String probabilityCol = hasProbabilityCol.getProbabilityCol();

			List<Feature> features = new ArrayList<>();

			for(int i = 0; i < categoricalLabel.size(); i++){
				Object value = categoricalLabel.getValue(i);

				OutputField probabilityField = ModelUtil.createProbabilityField(FieldName.create(probabilityCol + "(" + value + ")"), DataType.DOUBLE, value);

				result.add(probabilityField);

				features.add(new ContinuousFeature(encoder, probabilityField));
			}

			encoder.putFeatures(probabilityCol, features);
		}

		return result;
	}
}