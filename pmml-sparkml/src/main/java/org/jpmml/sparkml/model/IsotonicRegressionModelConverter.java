/*
 * Copyright (c) 2025 Villu Ruusmann
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

import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.IsotonicRegressionModel;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Field;
import org.dmg.pmml.LinearNorm;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutlierTreatmentMethod;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.jpmml.sparkml.ModelConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class IsotonicRegressionModelConverter extends ModelConverter<IsotonicRegressionModel> {

	public IsotonicRegressionModelConverter(IsotonicRegressionModel model){
		super(model);
	}

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.REGRESSION;
	}

	@Override
	public ContinuousLabel getLabel(SparkMLEncoder encoder){
		IsotonicRegressionModel model = getModel();

		String labelCol = model.getLabelCol();

		Feature feature = encoder.getOnlyFeature(labelCol);

		Field<?> field = encoder.toContinuous(feature);

		field.setDataType(DataType.DOUBLE);

		return new ContinuousLabel(field);
	}

	@Override
	public List<Feature> getFeatures(SparkMLEncoder encoder){
		IsotonicRegressionModel model = getModel();

		int featureIndex = model.getFeatureIndex();

		Vector boundaries = model.boundaries();
		Vector predictions = model.predictions();

		String featuresCol = model.getFeaturesCol();

		List<Feature> features = encoder.getFeatures(featuresCol);

		Feature feature = features.get(featureIndex);

		NormContinuous normContinuous = new NormContinuous(feature.getName(), null)
			.setOutlierTreatment(OutlierTreatmentMethod.AS_EXTREME_VALUES);

		for(int i = 0, length = boundaries.size(); i < length; i++){
			Double orig = boundaries.apply(i);
			Double norm = predictions.apply(i);

			normContinuous.addLinearNorms(new LinearNorm(orig, norm));
		}

		DerivedField derivedField = encoder.createDerivedField("isotonicRegression", OpType.CONTINUOUS, DataType.DOUBLE, normContinuous);

		return Collections.singletonList(new ContinuousFeature(encoder, derivedField));
	}

	@Override
	public Model encodeModel(Schema schema){
		IsotonicRegressionModel model = getModel();

		List<? extends Feature> features = schema.getFeatures();

		SchemaUtil.checkSize(1, features);

		return RegressionModelUtil.createRegression(features, Collections.singletonList(1d), null, RegressionModel.NormalizationMethod.NONE, schema);
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, Model pmmlModel, SparkMLEncoder encoder){
		IsotonicRegressionModel model = getModel();

		ScalarLabel scalarLabel = (ScalarLabel)label;

		String predictionCol = model.getPredictionCol();

		Boolean keepPredictionCol = (Boolean)getOption(HasPredictionModelOptions.OPTION_KEEP_PREDICTIONCOL, Boolean.TRUE);

		OutputField predictedOutputField = ModelUtil.createPredictedField(encoder.mapOnlyFieldName(predictionCol), OpType.CONTINUOUS, scalarLabel.getDataType());

		DerivedOutputField predictedField = encoder.createDerivedField(pmmlModel, predictedOutputField, keepPredictionCol);

		encoder.putOnlyFeature(predictionCol, new ContinuousFeature(encoder, predictedField));

		return Collections.emptyList();
	}
}