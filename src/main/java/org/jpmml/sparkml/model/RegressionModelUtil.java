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
import org.dmg.pmml.regression.CategoricalPredictor;
import org.dmg.pmml.regression.NumericPredictor;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;

public class RegressionModelUtil {

	private RegressionModelUtil(){
	}

	static
	public RegressionTable encodeRegressionTable(double intercept, Vector coefficients, Schema schema){
		RegressionTable regressionTable = new RegressionTable(intercept);

		List<Feature> features = schema.getFeatures();
		if(features.size() != coefficients.size()){
			throw new IllegalArgumentException();
		}

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			if(feature instanceof ContinuousFeature){
				ContinuousFeature continuousFeature = (ContinuousFeature)feature;

				NumericPredictor numericPredictor = new NumericPredictor()
					.setName(continuousFeature.getName())
					.setCoefficient(coefficients.apply(i));

				regressionTable.addNumericPredictors(numericPredictor);
			} else

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				String value = ValueUtil.formatValue(binaryFeature.getValue());

				CategoricalPredictor categoricalPredictor = new CategoricalPredictor()
					.setName(binaryFeature.getName())
					.setCoefficient(coefficients.apply(i))
					.setValue(value);

				regressionTable.addCategoricalPredictors(categoricalPredictor);
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		return regressionTable;
	}
}