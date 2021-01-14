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
import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.feature.StringIndexerModel;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Field;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.Value;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class StringIndexerModelConverter extends FeatureConverter<StringIndexerModel> {

	public StringIndexerModelConverter(StringIndexerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		StringIndexerModel transformer = getTransformer();

		Feature feature = encoder.getOnlyFeature(transformer.getInputCol());

		List<String> categories = new ArrayList<>();
		categories.addAll(Arrays.asList(transformer.labels()));

		String invalidCategory;

		DataType dataType = feature.getDataType();
		switch(dataType){
			case INTEGER:
			case FLOAT:
			case DOUBLE:
				invalidCategory = "-999";
				break;
			default:
				invalidCategory = "__unknown";
				break;
		}

		String handleInvalid = transformer.getHandleInvalid();

		Field<?> field = encoder.toCategorical(feature.getName(), categories);

		if(field instanceof DataField){
			DataField dataField = (DataField)field;

			InvalidValueDecorator invalidValueDecorator;
			MissingValueDecorator missingValueDecorator;

			switch(handleInvalid){
				case "keep":
					{
						invalidValueDecorator = new InvalidValueDecorator(InvalidValueTreatmentMethod.AS_IS, invalidCategory);

						PMMLUtil.addValues(dataField, Collections.singletonList(invalidCategory), Value.Property.INVALID);

						categories.add(invalidCategory);

						missingValueDecorator = new MissingValueDecorator(MissingValueTreatmentMethod.AS_VALUE, invalidCategory);

						encoder.addDecorator(dataField, missingValueDecorator);
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
						Apply setApply = PMMLUtil.createApply(PMMLFunctions.ISIN, feature.ref());

						for(String category : categories){
							setApply.addExpressions(PMMLUtil.createConstant(category, dataType));
						}

						categories.add(invalidCategory);

						Apply apply = PMMLUtil.createApply(PMMLFunctions.IF)
							.addExpressions(setApply)
							.addExpressions(feature.ref(), PMMLUtil.createConstant(invalidCategory, dataType));

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

		return Collections.singletonList(new CategoricalFeature(encoder, field, categories));
	}
}