/*
 * Copyright (c) 2022 Villu Ruusmann
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
package org.jpmml.sparkml.lightgbm;

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassificationModel;
import org.apache.spark.ml.param.shared.HasProbabilityCol;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.sparkml.ClassificationModelConverter;

public class LightGBMClassificationModelConverter extends ClassificationModelConverter<LightGBMClassificationModel> {

	public LightGBMClassificationModelConverter(LightGBMClassificationModel model){
		super(model);
	}

	@Override
	public int getNumberOfClasses(){
		int numberOfClasses = super.getNumberOfClasses();

		if(numberOfClasses == 1){
			return 2;
		}

		return numberOfClasses;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		LightGBMClassificationModel model = getTransformer();

		MiningModel miningModel = BoosterUtil.encodeModel(this, schema);

		RegressionModel regressionModel = (RegressionModel)MiningModelUtil.getFinalModel(miningModel);

		if(model instanceof HasProbabilityCol){
			regressionModel.setOutput(null);
		}

		return miningModel;
	}
}