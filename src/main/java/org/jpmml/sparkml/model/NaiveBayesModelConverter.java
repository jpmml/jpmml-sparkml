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

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.linalg.Vector;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.sparkml.ClassificationModelConverter;
import org.jpmml.sparkml.MatrixUtil;
import org.jpmml.sparkml.VectorUtil;

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

		try {
			double[] thresholds = model.getThresholds();

			for(int i = 0; i < thresholds.length; i++){
				double threshold = thresholds[i];

				if(threshold != 0d){
					throw new IllegalArgumentException("Non-zero thresholds are not supported");
				}
			}
		} catch(NoSuchElementException nsee){
			// Ignored
		}

		Vector pi = model.pi();
		Matrix theta = model.theta();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		MatrixUtil.checkRows(categoricalLabel.size(), theta);

		List<Double> intercepts = VectorUtil.toList(pi);

		scala.collection.Iterator<Vector> thetaRows = theta.rowIter();

		RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), null)
			.setNormalizationMethod(RegressionModel.NormalizationMethod.SOFTMAX);

		for(int i = 0; i < categoricalLabel.size(); i++){
			Object targetCategory = categoricalLabel.getValue(i);

			List<Feature> features = new ArrayList<>(schema.getFeatures());
			List<Double> coefficients = new ArrayList<>(VectorUtil.toList(thetaRows.next()));

			RegressionTableUtil.simplify(this, targetCategory, features, coefficients);

			RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(features, coefficients, intercepts.get(i))
				.setTargetCategory(targetCategory);

			regressionModel.addRegressionTables(regressionTable);
		}

		return regressionModel;
	}
}