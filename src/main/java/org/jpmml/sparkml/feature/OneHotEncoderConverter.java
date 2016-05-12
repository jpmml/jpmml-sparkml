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
import java.util.List;

import org.apache.spark.ml.feature.OneHotEncoder;
import org.jpmml.sparkml.BinaryFeature;
import org.jpmml.sparkml.Feature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;
import org.jpmml.sparkml.ListFeature;

public class OneHotEncoderConverter extends FeatureConverter<OneHotEncoder> {

	public OneHotEncoderConverter(OneHotEncoder transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		OneHotEncoder transformer = getTransformer();

		ListFeature inputFeature = (ListFeature)featureMapper.getOnlyFeature(transformer.getInputCol());

		List<String> values = inputFeature.getValues();

		Boolean dropLast = (Boolean)transformer.get(transformer.dropLast()).get();
		if(dropLast){
			values = values.subList(0, values.size() - 1);
		}

		List<Feature> result = new ArrayList<>();

		for(String value : values){
			Feature feature = new BinaryFeature(inputFeature.getName(), value);

			result.add(feature);
		}

		return result;
	}
}