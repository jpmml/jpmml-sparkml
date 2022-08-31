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

import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sparkml.RegressionModelConverter;

public class RandomForestRegressionModelConverter extends RegressionModelConverter<RandomForestRegressionModel> implements HasFeatureImportances, HasTreeOptions {

	public RandomForestRegressionModelConverter(RandomForestRegressionModel model){
		super(model);
	}

	@Override
	public Vector getFeatureImportances(){
		RandomForestRegressionModel model = getTransformer();

		return model.featureImportances();
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		List<TreeModel> treeModels = TreeModelUtil.encodeDecisionTreeEnsemble(this, schema);

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema.getLabel()))
			.setSegmentation(MiningModelUtil.createSegmentation(Segmentation.MultipleModelMethod.AVERAGE, Segmentation.MissingPredictionTreatment.RETURN_MISSING, treeModels));

		return miningModel;
	}
}