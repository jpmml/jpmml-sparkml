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
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.OpType;
import org.dmg.pmml.TypeDefinitionField;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.InvalidValueDecorator;
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

		List<String> labels = Arrays.asList(transformer.labels());

		Feature feature = encoder.getOnlyFeature(transformer.getInputCol());

		DataField dataField = encoder.toCategorical(feature.getName(), labels);

		InvalidValueTreatmentMethod invalidValueTreatmentMethod;

		String handleInvalid = transformer.getHandleInvalid();
		switch(handleInvalid){
			case "keep":
				invalidValueTreatmentMethod = InvalidValueTreatmentMethod.AS_IS;
				break;
			case "error":
				invalidValueTreatmentMethod = InvalidValueTreatmentMethod.RETURN_INVALID;
				break;
			default:
				throw new IllegalArgumentException(handleInvalid);
		}

		InvalidValueDecorator invalidValueDecorator = new InvalidValueDecorator()
			.setInvalidValueTreatment(invalidValueTreatmentMethod);

		encoder.addDecorator(dataField.getName(), invalidValueDecorator);

		TypeDefinitionField field = dataField;

		switch(handleInvalid){
			case "keep":
				Apply setApply = PMMLUtil.createApply("isIn", feature.ref());

				for(String label : labels){
					setApply.addExpressions(PMMLUtil.createConstant(label));
				}

				labels = new ArrayList<>(labels);
				labels.add(StringIndexerModelConverter.LABEL_UNKNOWN);

				Apply apply = PMMLUtil.createApply("if", setApply, feature.ref(), PMMLUtil.createConstant(StringIndexerModelConverter.LABEL_UNKNOWN));

				field = encoder.createDerivedField(FeatureUtil.createName("handleInvalid", feature), OpType.CATEGORICAL, feature.getDataType(), apply);
				break;
			default:
				break;
		}

		return Collections.<Feature>singletonList(new CategoricalFeature(encoder, field, labels));
	}

	private static final String LABEL_UNKNOWN = "__unknown";
}