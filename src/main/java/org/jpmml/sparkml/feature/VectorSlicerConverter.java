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

import org.apache.spark.ml.feature.VectorSlicer;
import org.jpmml.converter.Feature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;

public class VectorSlicerConverter extends FeatureConverter<VectorSlicer> {

	public VectorSlicerConverter(VectorSlicer transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		VectorSlicer transformer = getTransformer();

		String[] names = transformer.getNames();
		if(names != null && names.length > 0){
			throw new IllegalArgumentException();
		}

		List<Feature> inputFeatures = featureMapper.getFeatures(transformer.getInputCol());

		List<Feature> result = new ArrayList<>();

		int[] indices = transformer.getIndices();
		for(int i = 0; i < indices.length; i++){
			int index = indices[i];

			Feature inputFeature = inputFeatures.get(index);

			result.add(inputFeature);
		}

		return result;
	}
}