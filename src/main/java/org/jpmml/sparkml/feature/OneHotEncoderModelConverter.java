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

		String[] inputCols = transformer.getInputCols();

		boolean dropLast = transformer.getDropLast();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < inputCols.length; i++){
			CategoricalFeature categoricalFeature = (CategoricalFeature)encoder.getOnlyFeature(inputCols[i]);

			List<String> values = categoricalFeature.getValues();
			if(dropLast){
				values = values.subList(0, values.size() - 1);
			}

			// XXX
			List<BinaryFeature> binaryFeatures = (List)OneHotEncoderConverter.encodeFeature(encoder, categoricalFeature, values);

			result.add(new BinarizedCategoricalFeature(encoder, categoricalFeature.getName(), categoricalFeature.getDataType(), binaryFeatures));
		}

		return result;
	}
}