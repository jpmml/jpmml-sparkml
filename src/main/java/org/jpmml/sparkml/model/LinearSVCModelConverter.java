/*
 * Copyright (c) 2019 Villu Ruusmann
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

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.classification.LinearSVCModel;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.AbstractTransformation;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.Transformation;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.sparkml.ClassificationModelConverter;
import org.jpmml.sparkml.VectorUtil;

public class LinearSVCModelConverter extends ClassificationModelConverter<LinearSVCModel> implements HasRegressionOptions {

	public LinearSVCModelConverter(LinearSVCModel model){
		super(model);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		LinearSVCModel model = getTransformer();

		double threshold = model.getThreshold();

		List<Feature> features = new ArrayList<>(schema.getFeatures());
		List<Double> coefficients = new ArrayList<>(VectorUtil.toList(model.coefficients()));

		RegressionTableUtil.simplify(this, null, features, coefficients);

		Transformation transformation = new AbstractTransformation(){

			@Override
			public Expression createExpression(FieldRef fieldRef){
				return PMMLUtil.createApply(PMMLFunctions.IF)
					.addExpressions(PMMLUtil.createApply(PMMLFunctions.GREATERTHAN, fieldRef, PMMLUtil.createConstant(threshold)))
					.addExpressions(PMMLUtil.createConstant(1), PMMLUtil.createConstant(0));
			}
		};

		Schema segmentSchema = schema.toAnonymousRegressorSchema(DataType.DOUBLE).toEmptySchema();

		RegressionModel regressionModel = RegressionModelUtil.createRegression(features, coefficients, model.intercept(), RegressionModel.NormalizationMethod.NONE, segmentSchema)
			.setOutput(ModelUtil.createPredictedOutput(FieldName.create("margin"), OpType.CONTINUOUS, DataType.DOUBLE, transformation));

		return MiningModelUtil.createBinaryLogisticClassification(regressionModel, 1d, 0d, RegressionModel.NormalizationMethod.NONE, false, schema);
	}
}