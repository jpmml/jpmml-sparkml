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

import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.general_regression.GeneralRegressionModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.general_regression.GeneralRegressionModelUtil;
import org.jpmml.sparkml.RegressionModelConverter;
import org.jpmml.sparkml.VectorUtil;

public class GeneralizedLinearRegressionModelConverter extends RegressionModelConverter<GeneralizedLinearRegressionModel> {

	public GeneralizedLinearRegressionModelConverter(GeneralizedLinearRegressionModel model){
		super(model);
	}

	@Override
	public MiningFunction getMiningFunction(){
		GeneralizedLinearRegressionModel model = getTransformer();

		String family = model.getFamily();
		switch(family){
			case "binomial":
				return MiningFunction.CLASSIFICATION;
			default:
				return MiningFunction.REGRESSION;
		}
	}

	@Override
	public GeneralRegressionModel encodeModel(Schema schema){
		GeneralizedLinearRegressionModel model = getTransformer();

		String targetCategory = null;

		List<String> targetCategories = schema.getTargetCategories();
		if(targetCategories != null && targetCategories.size() > 0){

			if(targetCategories.size() != 2){
				throw new IllegalArgumentException();
			}

			targetCategory = targetCategories.get(1);
		}

		MiningFunction miningFunction = (targetCategory != null ? MiningFunction.CLASSIFICATION : MiningFunction.REGRESSION);

		GeneralRegressionModel generalRegressionModel = new GeneralRegressionModel(GeneralRegressionModel.ModelType.GENERALIZED_LINEAR, miningFunction, ModelUtil.createMiningSchema(schema), null, null, null)
			.setDistribution(parseFamily(model.getFamily()))
			.setLinkFunction(parseLink(model.getLink()));

		GeneralRegressionModelUtil.encodeRegressionTable(generalRegressionModel, schema.getFeatures(), model.intercept(), VectorUtil.toList(model.coefficients()), targetCategory);

		switch(miningFunction){
			case CLASSIFICATION:
				generalRegressionModel.setOutput(ModelUtil.createProbabilityOutput(schema));
				break;
			default:
				break;
		}

		return generalRegressionModel;
	}

	static
	private GeneralRegressionModel.Distribution parseFamily(String family){

		switch(family){
			case "binomial":
				return GeneralRegressionModel.Distribution.BINOMIAL;
			case "gamma":
				return GeneralRegressionModel.Distribution.GAMMA;
			case "gaussian":
				return GeneralRegressionModel.Distribution.NORMAL;
			case "poisson":
				return GeneralRegressionModel.Distribution.POISSON;
			default:
				throw new IllegalArgumentException(family);
		}
	}

	static
	private GeneralRegressionModel.LinkFunction parseLink(String link){

		switch(link){
			case "cloglog":
				return GeneralRegressionModel.LinkFunction.CLOGLOG;
			case "identity":
				return GeneralRegressionModel.LinkFunction.IDENTITY;
			case "log":
				return GeneralRegressionModel.LinkFunction.LOG;
			case "logit":
				return GeneralRegressionModel.LinkFunction.LOGIT;
			case "probit":
				return GeneralRegressionModel.LinkFunction.PROBIT;
			default:
				throw new IllegalArgumentException(link);
		}
	}
}