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
package org.jpmml.sparkml.model;

import java.util.List;

import org.apache.spark.ml.regression.GBTRegressionModel;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.Segmentation;
import org.dmg.pmml.TreeModel;
import org.jpmml.converter.FeatureSchema;
import org.jpmml.converter.MiningModelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.sparkml.ModelConverter;

public class GBTRegressionModelConverter extends ModelConverter<GBTRegressionModel> {

	public GBTRegressionModelConverter(GBTRegressionModel model){
		super(model);
	}

	@Override
	public MiningModel encodeModel(FeatureSchema schema){
		GBTRegressionModel model = getTransformer();

		List<TreeModel> treeModels = TreeModelUtil.encodeDecisionTreeEnsemble(model, schema);

		double[] weights = model.treeWeights();
		for(int i = 0; i < weights.length; i++){
			TreeModelUtil.scalePredictions(treeModels.get(i), weights[i]);
		}

		Segmentation segmentation = MiningModelUtil.createSegmentation(MultipleModelMethodType.SUM, treeModels);

		MiningModel miningModel = new MiningModel(MiningFunctionType.REGRESSION, ModelUtil.createMiningSchema(schema))
			.setSegmentation(segmentation);

		return miningModel;
	}
}