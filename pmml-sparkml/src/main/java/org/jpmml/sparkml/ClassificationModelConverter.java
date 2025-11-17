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

import org.apache.spark.ml.classification.ClassificationModel;
import org.apache.spark.ml.linalg.Vector;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.DerivedOutputField;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.IndexFeature;
import org.jpmml.converter.Label;
import org.jpmml.converter.LabelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.sparkml.model.HasPredictionModelOptions;

abstract
public class ClassificationModelConverter<T extends ClassificationModel<Vector, T>> extends PredictionModelConverter<T> {

	public ClassificationModelConverter(T model){
		super(model);
	}

	public int getNumberOfClasses(){
		T model = getModel();

		return model.numClasses();
	}

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.CLASSIFICATION;
	}

	@Override
	public void checkSchema(Schema schema){
		super.checkSchema(schema);

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

		SchemaUtil.checkSize(getNumberOfClasses(), categoricalLabel);
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, Model pmmlModel, SparkMLEncoder encoder){
		T model = getModel();

		CategoricalLabel categoricalLabel = (CategoricalLabel)label;

		List<Integer> categories = LabelUtil.createTargetCategories(categoricalLabel.size());

		String predictionCol = model.getPredictionCol();

		Boolean keepPredictionCol = (Boolean)getOption(HasPredictionModelOptions.OPTION_KEEP_PREDICTIONCOL, Boolean.TRUE);

		OutputField pmmlPredictedOutputField = ModelUtil.createPredictedField(FieldNameUtil.create("pmml", predictionCol), OpType.CATEGORICAL, categoricalLabel.getDataType())
			.setFinalResult(false);

		DerivedOutputField pmmlPredictedField = encoder.createDerivedField(pmmlModel, pmmlPredictedOutputField, keepPredictionCol);

		MapValues mapValues = ExpressionUtil.createMapValues(pmmlPredictedField.getName(), categoricalLabel.getValues(), categories)
			.setDataType(DataType.DOUBLE);

		OutputField predictedOutputField = new OutputField(encoder.mapOnlyFieldName(predictionCol), OpType.CONTINUOUS, DataType.DOUBLE)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(mapValues);

		DerivedOutputField predictedField = encoder.createDerivedField(pmmlModel, predictedOutputField, keepPredictionCol);

		encoder.putOnlyFeature(predictionCol, new IndexFeature(encoder, predictedField, categories));

		return Collections.emptyList();
	}
}