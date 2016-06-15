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
import org.dmg.pmml.DataType;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ListFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;
import scala.Option;

public class OneHotEncoderConverter extends FeatureConverter<OneHotEncoder> {

	public OneHotEncoderConverter(OneHotEncoder transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		OneHotEncoder transformer = getTransformer();

		ListFeature inputFeature = (ListFeature)featureMapper.getOnlyFeature(transformer.getInputCol());

		List<String> values = inputFeature.getValues();

		boolean dropLast = true;

		Option<Object> dropLastOption = transformer.get(transformer.dropLast());
		if(dropLastOption.isDefined()){
			dropLast = (Boolean)dropLastOption.get();
		} // End if

		if(dropLast){
			values = values.subList(0, values.size() - 1);
		}

		List<Feature> result = new ArrayList<>();

		for(String value : values){
			Feature feature = new BinaryFeature(inputFeature.getName(), DataType.STRING, value);

			result.add(feature);
		}

		return result;
	}
}