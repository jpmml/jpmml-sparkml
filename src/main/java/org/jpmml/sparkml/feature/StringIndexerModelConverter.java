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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.feature.StringIndexerModel;
import org.dmg.pmml.DataField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ListFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;

public class StringIndexerModelConverter extends FeatureConverter<StringIndexerModel> {

	public StringIndexerModelConverter(StringIndexerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		StringIndexerModel transformer = getTransformer();

		Feature inputFeature = featureMapper.getOnlyFeature(transformer.getInputCol());

		List<String> categories = Arrays.asList(transformer.labels());

		DataField dataField = featureMapper.toCategorical(inputFeature.getName(), categories);

		Feature feature = new ListFeature(dataField, categories);

		return Collections.singletonList(feature);
	}
}