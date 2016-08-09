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
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;

import org.apache.spark.ml.feature.VectorIndexerModel;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Row;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DOMUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ListFeature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;

public class VectorIndexerModelConverter extends FeatureConverter<VectorIndexerModel> {

	public VectorIndexerModelConverter(VectorIndexerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		VectorIndexerModel transformer = getTransformer();

		List<Feature> inputFeatures = featureMapper.getFeatures(transformer.getInputCol());

		int numFeatures = transformer.numFeatures();
		if(numFeatures != inputFeatures.size()){
			throw new IllegalArgumentException();
		}

		Map<Integer, Map<Double, Integer>> categoryMaps = transformer.javaCategoryMaps();

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < numFeatures; i++){
			Feature inputFeature = inputFeatures.get(i);

			ContinuousFeature feature;

			Map<Double, Integer> categoryMap = categoryMaps.get(i);
			if(categoryMap != null){
				List<String> categories = new ArrayList<>();
				List<String> values = new ArrayList<>();

				DocumentBuilder documentBuilder = DOMUtil.createDocumentBuilder();

				InlineTable inlineTable = new InlineTable();

				List<String> columns = Arrays.asList("input", "output");

				List<Map.Entry<Double, Integer>> entries = new ArrayList<>(categoryMap.entrySet());
				Collections.sort(entries, VectorIndexerModelConverter.COMPARATOR);

				for(Map.Entry<Double, Integer> entry : entries){
					String category = ValueUtil.formatValue(entry.getKey());

					categories.add(category);

					String value = ValueUtil.formatValue(entry.getValue());

					values.add(value);

					Row row = DOMUtil.createRow(documentBuilder, columns, Arrays.asList(category, value));

					inlineTable.addRows(row);
				}

				featureMapper.toCategorical(inputFeature.getName(), categories);

				MapValues mapValues = new MapValues()
					.addFieldColumnPairs(new FieldColumnPair(inputFeature.getName(), columns.get(0)))
					.setOutputColumn(columns.get(1))
					.setInlineTable(inlineTable);

				DerivedField derivedField = featureMapper.createDerivedField(formatName(transformer, i), OpType.CATEGORICAL, DataType.INTEGER, mapValues);

				feature = new ListFeature(derivedField, values);
			} else

			{
				feature = (ContinuousFeature)inputFeature;
			}

			result.add(feature);
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