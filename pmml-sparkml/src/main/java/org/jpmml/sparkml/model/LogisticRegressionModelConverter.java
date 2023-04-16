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
import org.dmg.pmml.Model;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.ProbabilisticClassificationModelConverter;

public class LogisticRegressionModelConverter extends ProbabilisticClassificationModelConverter<LogisticRegressionModel> implements HasRegressionTableOptions {

	public LogisticRegressionModelConverter(LogisticRegressionModel model){
		super(model);
	}

	@Override
	public Model encodeModel(Schema schema){
		LogisticRegressionModel model = getModel();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		if(categoricalLabel.size() == 2){
			Model linearModel = LinearModelUtil.createBinaryLogisticClassification(this, model.coefficients(), model.intercept(), schema)
				.setOutput(null);

			return linearModel;
		} else

		if(categoricalLabel.size() > 2){
			Model linearModel = LinearModelUtil.createSoftmaxClassification(this, model.coefficientMatrix(), model.interceptVector(), schema)
				.setOutput(null);

			return linearModel;
		} else

		{
			throw new IllegalArgumentException();
		}
	}
}