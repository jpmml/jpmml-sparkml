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

import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.sparkml.BinarizedCategoricalFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class OneHotEncoderModelConverter extends FeatureConverter<OneHotEncoderModel> {

	public OneHotEncoderModelConverter(OneHotEncoderModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		OneHotEncoderModel transformer = getTransformer();

		boolean dropLast = transformer.getDropLast();

		List<Feature> result = new ArrayList<>();

		String[] inputCols = transformer.getInputCols();
		for(String inputCol : inputCols){
			CategoricalFeature categoricalFeature = (CategoricalFeature)encoder.getOnlyFeature(inputCol);

			List<?> values = categoricalFeature.getValues();

			List<BinaryFeature> binaryFeatures = OneHotEncoderConverter.encodeFeature(encoder, categoricalFeature, values, dropLast);

			result.add(new BinarizedCategoricalFeature(encoder, categoricalFeature.getName(), categoricalFeature.getDataType(), binaryFeatures));
		}

		return result;
	}
}