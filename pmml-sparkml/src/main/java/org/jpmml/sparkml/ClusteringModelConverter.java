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

import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.shared.HasFeaturesCol;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.IndexFeature;
import org.jpmml.converter.Label;
import org.jpmml.converter.LabelUtil;
import org.jpmml.converter.ModelUtil;

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
	public List<Feature> getFeatures(SparkMLEncoder encoder){
		T model = getTransformer();

		String featuresCol = model.getFeaturesCol();

		return encoder.getFeatures(featuresCol);
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, org.dmg.pmml.Model pmmlModel, SparkMLEncoder encoder){
		T model = getTransformer();

		List<Integer> clusters = LabelUtil.createTargetCategories(getNumberOfClusters());

		String predictionCol = model.getPredictionCol();

		OutputField pmmlPredictedOutputField = ModelUtil.createPredictedField(FieldNameUtil.create("pmml", predictionCol), OpType.CATEGORICAL, DataType.STRING)
			.setFinalResult(false);

		DerivedOutputField pmmlPredictedField = encoder.createDerivedField(pmmlModel, pmmlPredictedOutputField, true);

		OutputField predictedOutputField = new OutputField(predictionCol, OpType.CATEGORICAL, DataType.INTEGER)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(new FieldRef(pmmlPredictedField));

		DerivedOutputField predictedField = encoder.createDerivedField(pmmlModel, predictedOutputField, true);

		encoder.putOnlyFeature(predictionCol, new IndexFeature(encoder, predictedField, clusters));

		return Collections.emptyList();
	}
}