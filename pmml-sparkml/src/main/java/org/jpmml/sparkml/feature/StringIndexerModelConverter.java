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
package org.jpmml.sparkml.feature;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.StringIndexerModel;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Field;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.sparkml.MultiFeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class StringIndexerModelConverter extends MultiFeatureConverter<StringIndexerModel> {

	public StringIndexerModelConverter(StringIndexerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		StringIndexerModel transformer = getTransformer();

		String handleInvalid = transformer.getHandleInvalid();
		String[][] labelsArray = transformer.labelsArray();

		InOutMode inputMode = getInputMode();

		List<Feature> result = new ArrayList<>();

		String[] inputCols = inputMode.getInputCols(transformer);
		for(int i = 0; i < inputCols.length; i++){
			String inputCol = inputCols[i];
			String[] labels = labelsArray[i];

			Feature feature = encoder.getOnlyFeature(inputCol);

			DataType dataType = feature.getDataType();

			List<String> categories = new ArrayList<>();
			categories.addAll(Arrays.asList(labels));

			String invalidCategory = getInvalidCategory(dataType);

			Field<?> field = encoder.toCategorical(feature.getName(), categories);

			if(field instanceof DataField){
				DataField dataField = (DataField)field;

				InvalidValueDecorator invalidValueDecorator;

				switch(handleInvalid){
					case "keep":
						{
							invalidValueDecorator = new InvalidValueDecorator(InvalidValueTreatmentMethod.AS_VALUE, invalidCategory);

							categories.add(invalidCategory);
						}
						break;
					case "error":
						{
							invalidValueDecorator = new InvalidValueDecorator(InvalidValueTreatmentMethod.RETURN_INVALID, null);
						}
						break;
					default:
						throw new IllegalArgumentException("Invalid value handling strategy " + handleInvalid + " is not supported");
				}

				encoder.addDecorator(dataField, invalidValueDecorator);
			} else

			if(field instanceof DerivedField){

				switch(handleInvalid){
					case "keep":
						{
							FieldRef fieldRef = feature.ref();

							Apply apply = ExpressionUtil.createApply(PMMLFunctions.IF,
								ExpressionUtil.createValueApply(fieldRef, dataType, categories),
								fieldRef,
								ExpressionUtil.createConstant(dataType, invalidCategory)
							);

							categories.add(invalidCategory);

							field = encoder.createDerivedField(FieldNameUtil.create("handleInvalid", feature), OpType.CATEGORICAL, dataType, apply);
						}
						break;
					case "error":
						{
							// Ignored: Assume that a DerivedField element can never return an erroneous field value
						}
						break;
					default:
						throw new IllegalArgumentException(handleInvalid);
				}
			} else

			{
				throw new IllegalArgumentException();
			}

			result.add(new CategoricalFeature(encoder, field, categories));
		}

		return result;
	}

	static
	public String getInvalidCategory(DataType dataType){

		switch(dataType){
			case INTEGER:
			case FLOAT:
			case DOUBLE:
				return "-999";
			default:
				return "__unknown";
		}
	}
}