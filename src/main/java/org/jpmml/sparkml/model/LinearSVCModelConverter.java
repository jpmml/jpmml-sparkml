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

import org.apache.spark.ml.classification.LinearSVCModel;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.Transformation;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sparkml.ClassificationModelConverter;

public class LinearSVCModelConverter extends ClassificationModelConverter<LinearSVCModel> implements HasRegressionTableOptions {

	public LinearSVCModelConverter(LinearSVCModel model){
		super(model);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		LinearSVCModel model = getTransformer();

		Transformation transformation = new Transformation(){

			@Override
			public FieldName getName(FieldName name){
				return FieldNameUtil.create("threshold", name);
			}

			@Override
			public Expression createExpression(FieldRef fieldRef){
				return PMMLUtil.createApply(PMMLFunctions.THRESHOLD)
					.addExpressions(fieldRef, PMMLUtil.createConstant(model.getThreshold()));
			}
		};

		Schema segmentSchema = schema.toAnonymousRegressorSchema(DataType.DOUBLE);

		Model linearModel = LinearModelUtil.createRegression(this, model.coefficients(), model.intercept(), segmentSchema)
			.setOutput(ModelUtil.createPredictedOutput(FieldName.create("margin"), OpType.CONTINUOUS, DataType.DOUBLE, transformation));

		return MiningModelUtil.createBinaryLogisticClassification(linearModel, 1d, 0d, RegressionModel.NormalizationMethod.NONE, false, schema);
	}
}