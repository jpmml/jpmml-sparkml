/*
 * Copyright (c) 2018 Villu Ruusmann
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
import java.util.List;
import java.util.Objects;

import com.google.common.collect.Iterables;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.dmg.pmml.DataType;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.DiscreteFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.sparkml.BinarizedCategoricalFeature;
import org.jpmml.sparkml.MultiFeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class OneHotEncoderModelConverter extends MultiFeatureConverter<OneHotEncoderModel> {

	public OneHotEncoderModelConverter(OneHotEncoderModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		OneHotEncoderModel transformer = getTransformer();

		boolean dropLast = transformer.getDropLast();
		String handleInvalid = transformer.getHandleInvalid();

		boolean keepInvalid = Objects.equals("keep", handleInvalid);

		InOutMode inputMode = getInputMode();

		List<Feature> result = new ArrayList<>();

		String[] inputCols = inputMode.getInputCols(transformer);
		for(String inputCol : inputCols){
			DiscreteFeature discreteFeature = (DiscreteFeature)encoder.getOnlyFeature(inputCol);

			DataType dataType = discreteFeature.getDataType();

			// XXX
			String invalidCategory = StringIndexerModelConverter.getInvalidCategory(dataType);

			List<?> values = discreteFeature.getValues();

			List<BinaryFeature> binaryFeatures = OneHotEncoderModelConverter.encodeFeature(encoder, discreteFeature, values);

			if(!dropLast && keepInvalid){
				BinaryFeature invalidCategoryFeature = new BinaryFeature(encoder, discreteFeature, invalidCategory);

				binaryFeatures.add(invalidCategoryFeature);
			} else

			if(dropLast && !keepInvalid){
				binaryFeatures = binaryFeatures.subList(0, binaryFeatures.size() - 1);
			} else

			{
				// Ignored: No-op
			}

			result.add(new BinarizedCategoricalFeature(encoder, discreteFeature, binaryFeatures));
		}

		return result;
	}

	@Override
	public void registerFeatures(SparkMLEncoder encoder){
		OneHotEncoderModel transformer = getTransformer();

		List<Feature> features = encodeFeatures(encoder);

		InOutMode outputMode = getOutputMode();

		if(outputMode == InOutMode.SINGLE){
			String outputCol = transformer.getOutputCol();

			BinarizedCategoricalFeature binarizedCategoricalFeature = (BinarizedCategoricalFeature)Iterables.getOnlyElement(features);

			encoder.putFeatures(outputCol, (List)binarizedCategoricalFeature.getBinaryFeatures());
		} else

		if(outputMode == InOutMode.MULTIPLE){
			String[] outputCols = transformer.getOutputCols();

			SchemaUtil.checkSize(outputCols.length, features);

			for(int i = 0; i < outputCols.length; i++){
				String outputCol = outputCols[i];
				Feature feature = features.get(i);

				BinarizedCategoricalFeature binarizedCategoricalFeature = (BinarizedCategoricalFeature)feature;

				encoder.putFeatures(outputCol, (List)binarizedCategoricalFeature.getBinaryFeatures());
			}
		}
	}

	static
	public List<BinaryFeature> encodeFeature(PMMLEncoder encoder, Feature feature, List<?> values){
		List<BinaryFeature> result = new ArrayList<>();

		for(Object value : values){
			result.add(new BinaryFeature(encoder, feature, value));
		}

		return result;
	}
}