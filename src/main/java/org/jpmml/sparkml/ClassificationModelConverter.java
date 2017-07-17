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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;

import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.shared.HasFeaturesCol;
import org.apache.spark.ml.param.shared.HasLabelCol;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.apache.spark.ml.param.shared.HasProbabilityCol;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.Row;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DOMUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLEncoder;

abstract
public class ClassificationModelConverter<T extends PredictionModel<Vector, T> & HasLabelCol & HasFeaturesCol & HasPredictionCol> extends ModelConverter<T> {

	public ClassificationModelConverter(T model){
		super(model);
	}

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.CLASSIFICATION;
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, SparkMLEncoder encoder){
		T model = getTransformer();

		CategoricalLabel categoricalLabel = (CategoricalLabel)label;

		List<OutputField> result = new ArrayList<>();

		HasPredictionCol hasPredictionCol = (HasPredictionCol)model;

		String predictionCol = hasPredictionCol.getPredictionCol();

		OutputField pmmlPredictedField = ModelUtil.createPredictedField(FieldName.create("pmml(" + predictionCol + ")"), categoricalLabel.getDataType(), OpType.CATEGORICAL);

		result.add(pmmlPredictedField);

		List<String> categories = new ArrayList<>();

		DocumentBuilder documentBuilder = DOMUtil.createDocumentBuilder();

		InlineTable inlineTable = new InlineTable();

		List<String> columns = Arrays.asList("input", "output");

		for(int i = 0; i < categoricalLabel.size(); i++){
			String value = categoricalLabel.getValue(i);
			String category = String.valueOf(i);

			categories.add(category);

			Row row = DOMUtil.createRow(documentBuilder, columns, Arrays.asList(value, category));

			inlineTable.addRows(row);
		}

		MapValues mapValues = new MapValues()
			.addFieldColumnPairs(new FieldColumnPair(pmmlPredictedField.getName(), columns.get(0)))
			.setOutputColumn(columns.get(1))
			.setInlineTable(inlineTable);

		final
		OutputField predictedField = new OutputField(FieldName.create(predictionCol), DataType.DOUBLE)
			.setOpType(OpType.CATEGORICAL)
			.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
			.setExpression(mapValues);

		result.add(predictedField);

		Feature feature = new CategoricalFeature(encoder, predictedField.getName(), predictedField.getDataType(), categories){

			@Override
			public ContinuousFeature toContinuousFeature(){
				PMMLEncoder encoder = ensureEncoder();

				return new ContinuousFeature(encoder, getName(), getDataType());
			}
		};

		encoder.putFeatures(predictionCol, Collections.<Feature>singletonList(feature));

		if(model instanceof HasProbabilityCol){
			HasProbabilityCol hasProbabilityCol = (HasProbabilityCol)model;

			String probabilityCol = hasProbabilityCol.getProbabilityCol();

			List<Feature> features = new ArrayList<>();

			for(int i = 0; i < categoricalLabel.size(); i++){
				String value = categoricalLabel.getValue(i);

				OutputField probabilityField = ModelUtil.createProbabilityField(FieldName.create(probabilityCol + "(" + value + ")"), DataType.DOUBLE, value);

				result.add(probabilityField);

				features.add(new ContinuousFeature(encoder, probabilityField.getName(), probabilityField.getDataType()));
			}

			encoder.putFeatures(probabilityCol, features);
		}

		return result;
	}
}