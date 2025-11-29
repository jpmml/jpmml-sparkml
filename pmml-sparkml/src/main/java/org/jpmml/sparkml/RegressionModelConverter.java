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
package org.jpmml.sparkml;

import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.shared.HasLabelCol;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.apache.spark.ml.regression.RegressionModel;
import org.dmg.pmml.Field;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.sparkml.model.HasPredictionModelOptions;

abstract
public class RegressionModelConverter<T extends RegressionModel<Vector, T>> extends PredictionModelConverter<T> {

	public RegressionModelConverter(T model){
		super(model);
	}

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.REGRESSION;
	}

	@Override
	public ScalarLabel getLabel(SparkMLEncoder encoder){
		return RegressionModelConverter.getLabel(this, encoder);
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, org.dmg.pmml.Model pmmlModel, SparkMLEncoder encoder){
		return RegressionModelConverter.registerPredictionOutputField(this, label, pmmlModel, encoder);
	}

	static
	public <T extends Model<T> & HasLabelCol & HasPredictionCol> ContinuousLabel getLabel(ModelConverter<T> converter, SparkMLEncoder encoder){
		T model = converter.getModel();

		String labelCol = model.getLabelCol();

		Feature feature = encoder.getOnlyFeature(labelCol);

		Field<?> field = encoder.toContinuous(feature);

		field.setDataType(converter.getDataType());

		return new ContinuousLabel(field);
	}

	static
	public <T extends Model<T> & HasPredictionCol> List<OutputField> registerPredictionOutputField(ModelConverter<T> converter, Label label, org.dmg.pmml.Model pmmlModel, SparkMLEncoder encoder){
		T model = converter.getModel();

		ScalarLabel scalarLabel = (ScalarLabel)label;

		String predictionCol = model.getPredictionCol();

		Boolean keepPredictionCol = (Boolean)converter.getOption(HasPredictionModelOptions.OPTION_KEEP_PREDICTIONCOL, Boolean.TRUE);

		OutputField predictedOutputField = ModelUtil.createPredictedField(encoder.mapOnlyFieldName(predictionCol), OpType.CONTINUOUS, scalarLabel.getDataType());

		DerivedOutputField predictedField = encoder.createDerivedField(pmmlModel, predictedOutputField, keepPredictionCol);

		encoder.putOnlyFeature(predictionCol, new ContinuousFeature(encoder, predictedField));

		return Collections.emptyList();
	}
}