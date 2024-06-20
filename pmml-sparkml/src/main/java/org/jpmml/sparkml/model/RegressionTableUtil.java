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
package org.jpmml.sparkml.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.stream.Collectors;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.sparkml.ModelConverter;

public class RegressionTableUtil {

	private RegressionTableUtil(){
	}

	static
	public <C extends ModelConverter<?> & HasRegressionTableOptions> void simplify(C converter, Object identifier, List<Feature> features, List<Double> coefficients){
		SchemaUtil.checkSize(coefficients.size(), features);

		Integer lookupThreshold = (Integer)converter.getOption(HasRegressionTableOptions.OPTION_LOOKUP_THRESHOLD, null);
		if(lookupThreshold == null){
			return;
		}

		Map<String, Long> countMap = features.stream()
			.filter(feature -> (feature instanceof BinaryFeature))
			.collect(Collectors.groupingBy(feature -> ((BinaryFeature)feature).getName(), Collectors.counting()));

		Collection<? extends Map.Entry<String, Long>> entries = countMap.entrySet();
		for(Map.Entry<String, Long> entry : entries){

			if(entry.getValue() < lookupThreshold){
				continue;
			}

			createMapValues(entry.getKey(), identifier, features, coefficients);
		}
	}

	static
	private MapValues createMapValues(String name, Object identifier, List<Feature> features, List<Double> coefficients){
		ListIterator<Feature> featureIt = features.listIterator();
		ListIterator<Double> coefficientIt = coefficients.listIterator();

		PMMLEncoder encoder = null;

		List<Object> inputValues = new ArrayList<>();
		List<Double> outputValues = new ArrayList<>();

		while(featureIt.hasNext()){
			Feature feature = featureIt.next();
			Double coefficient = coefficientIt.next();

			if(!(feature instanceof BinaryFeature)){
				continue;
			}

			BinaryFeature binaryFeature = (BinaryFeature)feature;
			if(!(name).equals(binaryFeature.getName())){
				continue;
			}

			featureIt.remove();
			coefficientIt.remove();

			if(encoder == null){
				encoder = binaryFeature.getEncoder();
			}

			inputValues.add(binaryFeature.getValue());
			outputValues.add(coefficient);
		}

		MapValues mapValues = ExpressionUtil.createMapValues(name, inputValues, outputValues)
			.setDefaultValue(0d)
			.setDataType(DataType.DOUBLE);

		DerivedField derivedField = encoder.createDerivedField(identifier != null ? FieldNameUtil.create("lookup", name, identifier) : FieldNameUtil.create("lookup", name), OpType.CONTINUOUS, DataType.DOUBLE, mapValues);

		featureIt.add(new ContinuousFeature(encoder, derivedField));
		coefficientIt.add(1d);

		return mapValues;
	}
}