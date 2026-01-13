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

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.general_regression.GeneralRegressionModel;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ExceptionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.general_regression.GeneralRegressionModelUtil;
import org.jpmml.sparkml.ClassificationModelConverter;
import org.jpmml.sparkml.RegressionModelConverter;
import org.jpmml.sparkml.SparkMLEncoder;
import org.jpmml.sparkml.SparkMLException;
import org.jpmml.sparkml.VectorUtil;

public class GeneralizedLinearRegressionModelConverter extends RegressionModelConverter<GeneralizedLinearRegressionModel> implements HasRegressionTableOptions {

	public GeneralizedLinearRegressionModelConverter(GeneralizedLinearRegressionModel model){
		super(model);
	}

	@Override
	public MiningFunction getMiningFunction(){
		GeneralizedLinearRegressionModel model = getModel();

		String family = model.getFamily();
		switch(family){
			case "binomial":
				return MiningFunction.CLASSIFICATION;
			default:
				return MiningFunction.REGRESSION;
		}
	}

	@Override
	public ScalarLabel getLabel(SparkMLEncoder encoder){
		GeneralizedLinearRegressionModel model = getModel();

		String family = model.getFamily();
		switch(family){
			case "binomial":
				return ClassificationModelConverter.getLabel(this, encoder);
			default:
				return RegressionModelConverter.getLabel(this, encoder);
		}
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, Model pmmlModel, SparkMLEncoder encoder){
		GeneralizedLinearRegressionModel model = getModel();

		List<OutputField> result = super.registerOutputFields(label, pmmlModel, encoder);

		MiningFunction miningFunction = getMiningFunction();
		switch(miningFunction){
			case CLASSIFICATION:
				CategoricalLabel categoricalLabel = (CategoricalLabel)label;

				result = new ArrayList<>(result);
				result.addAll(ModelUtil.createProbabilityFields(DataType.DOUBLE, categoricalLabel.getValues()));
				break;
			default:
				break;
		}

		return result;
	}

	@Override
	public GeneralRegressionModel encodeModel(Schema schema){
		GeneralizedLinearRegressionModel model = getModel();

		List<? extends Feature> features = schema.getFeatures();

		Object targetCategory = null;

		MiningFunction miningFunction = getMiningFunction();
		switch(miningFunction){
			case CLASSIFICATION:
				CategoricalLabel categoricalLabel = schema.requireCategoricalLabel()
					.expectCardinality(2);

				targetCategory = categoricalLabel.getValue(1);
				break;
			default:
				break;
		}

		List<Double> featureCoefficients = VectorUtil.toList(model.coefficients());

		GeneralRegressionModel generalRegressionModel = new GeneralRegressionModel(GeneralRegressionModel.ModelType.GENERALIZED_LINEAR, miningFunction, ModelUtil.createMiningSchema(schema), null, null, null)
			.setDistribution(parseFamily(model.getFamily()))
			.setLinkFunction(parseLinkFunction(model.getLink()))
			.setLinkParameter(parseLinkParameter(model.getLink()));

		GeneralRegressionModelUtil.encodeRegressionTable(generalRegressionModel, features, featureCoefficients, model.intercept(), targetCategory);

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
				throw new SparkMLException("Distribution family " + ExceptionUtil.formatParameter(family) + " is not supported");
		}
	}

	static
	private GeneralRegressionModel.LinkFunction parseLinkFunction(String link){

		switch(link){
			case "cloglog":
				return GeneralRegressionModel.LinkFunction.CLOGLOG;
			case "identity":
				return GeneralRegressionModel.LinkFunction.IDENTITY;
			case "inverse":
				return GeneralRegressionModel.LinkFunction.POWER;
			case "log":
				return GeneralRegressionModel.LinkFunction.LOG;
			case "logit":
				return GeneralRegressionModel.LinkFunction.LOGIT;
			case "probit":
				return GeneralRegressionModel.LinkFunction.PROBIT;
			case "sqrt":
				return GeneralRegressionModel.LinkFunction.POWER;
			default:
				throw new SparkMLException("Link function " + ExceptionUtil.formatParameter(link) + " is not supported");
		}
	}

	static
	private Double parseLinkParameter(String link){

		switch(link){
			case "inverse":
				return -1d;
			case "sqrt":
				return (1d / 2d);
			default:
				return null;
		}
	}
}