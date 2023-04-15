/*
 * Copyright (c) 2017 Villu Ruusmann
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
package org.jpmml.sparkml.xgboost;

import java.util.Collections;
import java.util.List;

import ml.dmlc.xgboost4j.scala.Booster;
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.PredictionModelConverter;
import org.jpmml.sparkml.SparkMLEncoder;
import org.jpmml.sparkml.model.HasPredictionModelOptions;
import org.jpmml.xgboost.HasXGBoostOptions;

public class XGBoostRegressionModelConverter extends PredictionModelConverter<XGBoostRegressionModel> implements HasXGBoostOptions {

	public XGBoostRegressionModelConverter(XGBoostRegressionModel model){
		super(model);
	}

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.REGRESSION;
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		XGBoostRegressionModel model = getTransformer();

		Booster booster = model.nativeBooster();

		return BoosterUtil.encodeBooster(this, booster, schema);
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, Model pmmlModel, SparkMLEncoder encoder){
		XGBoostRegressionModel model = getTransformer();

		ScalarLabel scalarLabel = (ScalarLabel)label;

		String predictionCol = model.getPredictionCol();

		Boolean keepPredictionCol = (Boolean)getOption(HasPredictionModelOptions.OPTION_KEEP_PREDICTIONCOL, Boolean.TRUE);

		OutputField predictedOutputField = ModelUtil.createPredictedField(predictionCol, OpType.CONTINUOUS, scalarLabel.getDataType());

		DerivedOutputField predictedField = encoder.createDerivedField(pmmlModel, predictedOutputField, keepPredictionCol);

		encoder.putOnlyFeature(predictionCol, new ContinuousFeature(encoder, predictedField));

		return Collections.emptyList();
	}
}