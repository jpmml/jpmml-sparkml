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

import com.google.common.primitives.Doubles;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.linalg.Vector;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sparkml.ClassificationModelConverter;

public class GBTClassificationModelConverter extends ClassificationModelConverter<GBTClassificationModel> implements HasFeatureImportances, HasTreeOptions {

	public GBTClassificationModelConverter(GBTClassificationModel model){
		super(model);
	}

	@Override
	public Vector getFeatureImportances(){
		GBTClassificationModel model = getTransformer();

		return model.featureImportances();
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		GBTClassificationModel model = getTransformer();

		String lossType = model.getLossType();
		switch(lossType){
			case "logistic":
				break;
			default:
				throw new IllegalArgumentException("Loss function " + lossType + " is not supported");
		}

		Schema segmentSchema = schema.toAnonymousRegressorSchema(DataType.DOUBLE);

		List<TreeModel> treeModels = TreeModelUtil.encodeDecisionTreeEnsemble(this, segmentSchema);

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(segmentSchema.getLabel()))
			.setSegmentation(MiningModelUtil.createSegmentation(Segmentation.MultipleModelMethod.WEIGHTED_SUM, Segmentation.MissingPredictionTreatment.RETURN_MISSING, treeModels, Doubles.asList(model.treeWeights())))
			.setOutput(ModelUtil.createPredictedOutput("gbtValue", OpType.CONTINUOUS, DataType.DOUBLE));

		return MiningModelUtil.createBinaryLogisticClassification(miningModel, 2d, 0d, RegressionModel.NormalizationMethod.LOGIT, false, schema);
	}
}