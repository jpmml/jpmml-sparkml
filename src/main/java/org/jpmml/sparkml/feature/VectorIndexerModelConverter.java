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
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import org.apache.spark.ml.feature.VectorIndexerModel;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class VectorIndexerModelConverter extends FeatureConverter<VectorIndexerModel> {

	public VectorIndexerModelConverter(VectorIndexerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		VectorIndexerModel transformer = getTransformer();

		int numFeatures = transformer.numFeatures();

		List<Feature> features = encoder.getFeatures(transformer.getInputCol());

		SchemaUtil.checkSize(numFeatures, features);

		Map<Integer, Map<Double, Integer>> categoryMaps = transformer.javaCategoryMaps();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < numFeatures; i++){
			Feature feature = features.get(i);

			Map<Double, Integer> categoryMap = categoryMaps.get(i);
			if(categoryMap != null){
				List<Double> categories = new ArrayList<>();
				List<Integer> values = new ArrayList<>();

				List<Map.Entry<Double, Integer>> entries = new ArrayList<>(categoryMap.entrySet());
				Collections.sort(entries, VectorIndexerModelConverter.COMPARATOR);

				for(Map.Entry<Double, Integer> entry : entries){
					Double category = entry.getKey();
					Integer value = entry.getValue();

					categories.add(category);
					values.add(value);
				}

				encoder.toCategorical(feature.getName(), categories);

				MapValues mapValues = PMMLUtil.createMapValues(feature.getName(), categories, values)
					.setDataType(DataType.INTEGER);

				DerivedField derivedField = encoder.createDerivedField(formatName(transformer, i), OpType.CATEGORICAL, DataType.INTEGER, mapValues);

				result.add(new CategoricalFeature(encoder, derivedField, values));
			} else

			{
				result.add((ContinuousFeature)feature);
			}
		}

		return result;
	}

	private static final Comparator<Map.Entry<Double, Integer>> COMPARATOR = new Comparator<Map.Entry<Double, Integer>>(){

		@Override
		public int compare(Map.Entry<Double, Integer> left, Map.Entry<Double, Integer> right){
			return (left.getValue()).compareTo(right.getValue());
		}
	};
}