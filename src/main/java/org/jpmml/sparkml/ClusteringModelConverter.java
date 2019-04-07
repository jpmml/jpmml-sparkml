/*
 * Copyright (c) 2017 Villu Ruusmann
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

import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.shared.HasFeaturesCol;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.LabelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLEncoder;

abstract
public class ClusteringModelConverter<T extends Model<T> & HasFeaturesCol & HasPredictionCol> extends ModelConverter<T> {

	public ClusteringModelConverter(T model){
		super(model);
	}

	abstract
	public int getNumberOfClusters();

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.CLUSTERING;
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, SparkMLEncoder encoder){
		T model = getTransformer();

		List<OutputField> result = new ArrayList<>();

		String predictionCol = model.getPredictionCol();

		OutputField pmmlPredictedField = ModelUtil.createPredictedField(FieldName.create("pmml(" + predictionCol + ")"), DataType.STRING, OpType.CATEGORICAL)
			.setFinalResult(false);

		result.add(pmmlPredictedField);

		OutputField predictedField = new OutputField(FieldName.create(predictionCol), DataType.INTEGER)
			.setOpType(OpType.CATEGORICAL)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(new FieldRef(pmmlPredictedField.getName()));

		result.add(predictedField);

		List<Integer> clusters = LabelUtil.createTargetCategories(getNumberOfClusters());

		Feature feature = new CategoricalFeature(encoder, predictedField, clusters){

			@Override
			public ContinuousFeature toContinuousFeature(){
				PMMLEncoder encoder = ensureEncoder();

				return new ContinuousFeature(encoder, predictedField);
			}
		};

		encoder.putOnlyFeature(predictionCol, feature);

		return result;
	}
}