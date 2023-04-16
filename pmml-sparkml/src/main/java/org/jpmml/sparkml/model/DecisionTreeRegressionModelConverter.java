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

import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.RegressionModelConverter;

public class DecisionTreeRegressionModelConverter extends RegressionModelConverter<DecisionTreeRegressionModel> implements HasFeatureImportances, HasTreeOptions {

	public DecisionTreeRegressionModelConverter(DecisionTreeRegressionModel model){
		super(model);
	}

	@Override
	public Vector getFeatureImportances(){
		DecisionTreeRegressionModel model = getModel();

		return model.featureImportances();
	}

	@Override
	public TreeModel encodeModel(Schema schema){
		return TreeModelUtil.encodeDecisionTree(this, schema);
	}
}