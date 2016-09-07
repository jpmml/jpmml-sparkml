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

import org.apache.spark.ml.classification.GBTClassificationModel;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sparkml.ClassificationModelConverter;

public class GBTClassificationModelConverter extends ClassificationModelConverter<GBTClassificationModel> {

	public GBTClassificationModelConverter(GBTClassificationModel model){
		super(model);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		GBTClassificationModel model = getTransformer();

		Schema segmentSchema = schema.toAnonymousSchema();

		List<TreeModel> treeModels = TreeModelUtil.encodeDecisionTreeEnsemble(model, model.treeWeights(), segmentSchema);

		Output output = encodeOutput();

		MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(segmentSchema))
			.setSegmentation(MiningModelUtil.createSegmentation(Segmentation.MultipleModelMethod.SUM, treeModels))
			.setOutput(output);

		return MiningModelUtil.createBinaryLogisticClassification(schema, miningModel, 1000d, false);
	}

	static
	private Output encodeOutput(){
		OutputField gbtValue = new OutputField(FieldName.create("gbtValue"), DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.setResultFeature(ResultFeature.PREDICTED_VALUE)
			.setFinalResult(false);

		OutputField binarizedGbtValue = new OutputField(FieldName.create("binarizedGbtValue"), DataType.DOUBLE)
			.setOpType(OpType.CONTINUOUS)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setFinalResult(false)
			.setExpression(PMMLUtil.createApply("if", PMMLUtil.createApply("greaterThan", new FieldRef(gbtValue.getName()), PMMLUtil.createConstant(0d)), PMMLUtil.createConstant(-1d), PMMLUtil.createConstant(1d)));

		Output output = new Output()
			.addOutputFields(gbtValue, binarizedGbtValue);

		return output;
	}
}