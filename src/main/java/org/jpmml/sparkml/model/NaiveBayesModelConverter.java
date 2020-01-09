/*
 * Copyright (c) 2018 Villu Ruusmann
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

import org.apache.spark.ml.classification.NaiveBayesModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.ClassificationModelConverter;

public class NaiveBayesModelConverter extends ClassificationModelConverter<NaiveBayesModel> implements HasRegressionOptions {

	public NaiveBayesModelConverter(NaiveBayesModel model){
		super(model);
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		NaiveBayesModel model = getTransformer();

		String modelType = model.getModelType();
		switch(modelType){
			case "multinomial":
				break;
			default:
				throw new IllegalArgumentException("Model type " + modelType + " is not supported");
		}

		if(model.isSet(model.thresholds())){
			double[] thresholds = model.getThresholds();

			for(int i = 0; i < thresholds.length; i++){
				double threshold = thresholds[i];

				if(threshold != 0d){
					throw new IllegalArgumentException("Non-zero thresholds are not supported");
				}
			}
		}

		RegressionModel regressionModel = LinearModelUtil.createSoftmaxClassification(this, model.theta(), model.pi(), schema)
			.setOutput(null);

		return regressionModel;
	}
}