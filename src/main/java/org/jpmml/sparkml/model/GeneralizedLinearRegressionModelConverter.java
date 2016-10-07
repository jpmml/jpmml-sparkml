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

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.general_regression.CovariateList;
import org.dmg.pmml.general_regression.FactorList;
import org.dmg.pmml.general_regression.GeneralRegressionModel;
import org.dmg.pmml.general_regression.PCell;
import org.dmg.pmml.general_regression.PPCell;
import org.dmg.pmml.general_regression.PPMatrix;
import org.dmg.pmml.general_regression.ParamMatrix;
import org.dmg.pmml.general_regression.Parameter;
import org.dmg.pmml.general_regression.ParameterList;
import org.dmg.pmml.general_regression.Predictor;
import org.dmg.pmml.general_regression.PredictorList;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.RegressionModelConverter;

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

		double intercept = model.intercept();
		Vector coefficients = model.coefficients();

		List<Feature> features = schema.getFeatures();
		if(features.size() != coefficients.size()){
			throw new IllegalArgumentException();
		}

		String targetCategory = null;

		List<String> targetCategories = schema.getTargetCategories();
		if(targetCategories != null && targetCategories.size() > 0){

			if(targetCategories.size() != 2){
				throw new IllegalArgumentException();
			}

			targetCategory = targetCategories.get(1);
		}

		ParameterList parameterList = new ParameterList();

		PPMatrix ppMatrix = new PPMatrix();

		ParamMatrix paramMatrix = new ParamMatrix();

		if(!ValueUtil.isZero(intercept)){
			Parameter parameter = new Parameter("p0")
				.setLabel("(intercept)");

			parameterList.addParameters(parameter);

			PCell pCell = new PCell(parameter.getName(), intercept)
				.setTargetCategory(targetCategory);

			paramMatrix.addPCells(pCell);
		}

		Set<FieldName> covariates = new LinkedHashSet<>();

		Set<FieldName> factors = new LinkedHashSet<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			Parameter parameter = new Parameter("p" + String.valueOf(i + 1));

			parameterList.addParameters(parameter);

			PPCell ppCell;

			if(feature instanceof ContinuousFeature){
				ContinuousFeature continuousFeature = (ContinuousFeature)feature;

				covariates.add(continuousFeature.getName());

				ppCell = new PPCell("1", continuousFeature.getName(), parameter.getName());
			} else

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				factors.add(binaryFeature.getName());

				ppCell = new PPCell(binaryFeature.getValue(), binaryFeature.getName(), parameter.getName());
			} else

			{
				throw new IllegalArgumentException();
			}

			ppMatrix.addPPCells(ppCell);

			PCell pCell = new PCell(parameter.getName(), coefficients.apply(i))
				.setTargetCategory(targetCategory);

			paramMatrix.addPCells(pCell);
		}

		MiningFunction miningFunction = (targetCategory != null ? MiningFunction.CLASSIFICATION : MiningFunction.REGRESSION);

		GeneralRegressionModel generalRegressionModel = new GeneralRegressionModel(GeneralRegressionModel.ModelType.GENERALIZED_LINEAR, miningFunction, ModelUtil.createMiningSchema(schema), parameterList, ppMatrix, paramMatrix)
			.setDistribution(parseFamily(model.getFamily()))
			.setLinkFunction(parseLink(model.getLink()))
			.setCovariateList(createPredictorList(new CovariateList(), covariates))
			.setFactorList(createPredictorList(new FactorList(), factors));

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

	static
	private <L extends PredictorList> L createPredictorList(L predictorList, Set<FieldName> names){

		if(names.isEmpty()){
			return null;
		}

		List<Predictor> predictors = predictorList.getPredictors();

		for(FieldName name : names){
			Predictor predictor = new Predictor(name);

			predictors.add(predictor);
		}

		return predictorList;
	}
}