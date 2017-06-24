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

import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.sparkml.ClassificationModelConverter;
import org.jpmml.sparkml.VectorUtil;

public class LogisticRegressionModelConverter extends ClassificationModelConverter<LogisticRegressionModel> {

	public LogisticRegressionModelConverter(LogisticRegressionModel model){
		super(model);
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		LogisticRegressionModel model = getTransformer();

		RegressionModel regressionModel = RegressionModelUtil.createBinaryLogisticClassification(schema.getFeatures(), VectorUtil.toList(model.coefficients()), model.intercept(), RegressionModel.NormalizationMethod.LOGIT, true, schema)
			.setOutput(null);

		return regressionModel;
	}
}