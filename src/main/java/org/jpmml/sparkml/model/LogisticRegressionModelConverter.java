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

import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.RegressionModel;
import org.dmg.pmml.RegressionNormalizationMethodType;
import org.dmg.pmml.RegressionTable;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.ModelConverter;

public class LogisticRegressionModelConverter extends ModelConverter<LogisticRegressionModel> {

	public LogisticRegressionModelConverter(LogisticRegressionModel model){
		super(model);
	}

	@Override
	public RegressionModel encodeModel(Schema schema){
		LogisticRegressionModel model = getTransformer();

		List<String> targetCategories = schema.getTargetCategories();
		if(targetCategories.size() != 2){
			throw new IllegalArgumentException();
		}

		RegressionTable activeRegressionTable = RegressionModelUtil.encodeRegressionTable(model.intercept(), model.coefficients(), schema)
			.setTargetCategory(targetCategories.get(1));

		RegressionTable passiveRegressionTable = new RegressionTable(0d)
			.setTargetCategory(targetCategories.get(0));

		RegressionModel regressionModel = new RegressionModel(MiningFunctionType.CLASSIFICATION, ModelUtil.createMiningSchema(schema), null)
			.setNormalizationMethod(RegressionNormalizationMethodType.SOFTMAX)
			.addRegressionTables(activeRegressionTable, passiveRegressionTable)
			.setOutput(ModelUtil.createProbabilityOutput(schema));

		return regressionModel;
	}
}