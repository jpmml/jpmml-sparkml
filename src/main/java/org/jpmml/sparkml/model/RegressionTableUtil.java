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

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.stream.Collectors;

import javax.xml.parsers.DocumentBuilder;

import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.InlineTable;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Row;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DOMUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.ModelConverter;

public class RegressionTableUtil {

	private RegressionTableUtil(){
	}

	static
	public <C extends ModelConverter<?> & HasRegressionOptions> void simplify(C converter, String identifier, List<Feature> features, List<Double> coefficients){

		if(features.size() != coefficients.size()){
			throw new IllegalArgumentException();
		}

		Integer lookupThreshold = (Integer)converter.getOption(HasRegressionOptions.OPTION_LOOKUP_THRESHOLD, null);
		if(lookupThreshold == null){
			return;
		}

		Map<FieldName, Long> countMap = features.stream()
			.filter(feature -> (feature instanceof BinaryFeature))
			.collect(Collectors.groupingBy(feature -> ((BinaryFeature)feature).getName(), Collectors.counting()));

		Collection<? extends Map.Entry<FieldName, Long>> entries = countMap.entrySet();
		for(Map.Entry<FieldName, Long> entry : entries){

			if(entry.getValue() < lookupThreshold){
				continue;
			}

			createMapValues(entry.getKey(), identifier, features, coefficients);
		}
	}

	static
	private MapValues createMapValues(FieldName name, String identifier, List<Feature> features, List<Double> coefficients){
		DocumentBuilder documentBuilder = DOMUtil.createDocumentBuilder();

		InlineTable inlineTable = new InlineTable();

		List<String> columns = Arrays.asList("input", "output");

		ListIterator<Feature> featureIt = features.listIterator();
		ListIterator<Double> coefficientIt = coefficients.listIterator();

		PMMLEncoder encoder = null;

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

			Row row = DOMUtil.createRow(documentBuilder, columns, Arrays.asList(binaryFeature.getValue(), coefficient));

			inlineTable.addRows(row);
		}

		MapValues mapValues = new MapValues()
			.addFieldColumnPairs(new FieldColumnPair(name, columns.get(0)))
			.setOutputColumn(columns.get(1))
			.setInlineTable(inlineTable)
			.setDefaultValue(ValueUtil.formatValue(0d));

		DerivedField derivedField = encoder.createDerivedField(FieldName.create("lookup(" + name.getValue() + (identifier != null ? (", " + identifier) : "") + ")"), OpType.CONTINUOUS, DataType.DOUBLE, mapValues);

		featureIt.add(new ContinuousFeature(encoder, derivedField));
		coefficientIt.add(1d);

		return mapValues;
	}
}