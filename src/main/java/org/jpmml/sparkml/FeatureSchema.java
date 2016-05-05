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
package org.jpmml.sparkml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import com.google.common.base.Predicate;
import com.google.common.collect.Iterables;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Value;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;

public class FeatureSchema extends Schema {

	private List<Feature> features = null;


	public FeatureSchema(FieldName targetField, List<String> targetCategories, List<Feature> features){
		super(targetField, targetCategories, extractActiveFields(features));

		setFeatures(features);
	}

	public DataDictionary encodeDataDictionary(){
		DataDictionary dataDictionary = new DataDictionary();

		{
			FieldName targetField = getTargetField();

			DataField dataField;

			List<String> targetCategories = getTargetCategories();
			if(targetCategories != null){
				dataField = new DataField(targetField, OpType.CATEGORICAL, DataType.STRING);

				List<Value> values = dataField.getValues();
				values.addAll(PMMLUtil.createValues(targetCategories));
			} else

			{
				dataField = new DataField(targetField, OpType.CONTINUOUS, DataType.DOUBLE);
			}

			dataDictionary.addDataFields(dataField);
		}

		List<FieldName> activeFields = getActiveFields();
		for(FieldName activeField : activeFields){
			DataField dataField;

			OpType opType = getFeatureType(activeField);
			switch(opType){
				case CONTINUOUS:
					dataField = new DataField(activeField, OpType.CONTINUOUS, DataType.DOUBLE);
					break;
				case CATEGORICAL:
					List<String> categories = new ArrayList<>(getFeatureCategories(activeField));

					DataType dataType = ValueUtil.getDataType(categories);
					switch(dataType){
						case INTEGER:
						case FLOAT:
						case DOUBLE:
							Comparator<String> comparator = new Comparator<String>(){

								@Override
								public int compare(String left, String right){
									return Double.compare(Double.parseDouble(left), Double.parseDouble(right));
								}
							};
							Collections.sort(categories, comparator);
							break;
						default:
							break;
					}

					dataField = new DataField(activeField, OpType.CATEGORICAL, dataType);

					List<Value> values = dataField.getValues();
					values.addAll(PMMLUtil.createValues(categories));
					break;
				default:
					throw new IllegalArgumentException();
			}

			dataDictionary.addDataFields(dataField);
		}

		return dataDictionary;
	}

	@Override
	public List<FieldName> getActiveFields(){
		return super.getActiveFields();
	}

	public OpType getFeatureType(FieldName name){
		Set<OpType> opTypes = new HashSet<>();

		Iterable<Feature> features = getFeatures(name);
		for(Feature feature : features){

			if(feature instanceof ContinuousFeature){
				opTypes.add(OpType.CONTINUOUS);
			} else

			if(feature instanceof CategoricalFeature){
				opTypes.add(OpType.CATEGORICAL);
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		return Iterables.getOnlyElement(opTypes);
	}

	public List<String> getFeatureCategories(FieldName name){
		List<String> result = new ArrayList<>();

		Iterable<Feature> features = getFeatures(name);
		for(Feature feature : features){
			CategoricalFeature<?> categoricalFeature = (CategoricalFeature<?>)feature;

			result.add((String)categoricalFeature.getValue());
		}

		return result;
	}

	private Iterable<Feature> getFeatures(final FieldName name){
		Predicate<Feature> predicate = new Predicate<Feature>(){

			@Override
			public boolean apply(Feature feature){
				return (feature.getName()).equals(name);
			}
		};

		return Iterables.filter(getFeatures(), predicate);
	}

	public Feature getFeature(int index){
		List<Feature> features = getFeatures();

		return features.get(index);
	}

	public List<Feature> getFeatures(){
		return this.features;
	}

	private void setFeatures(List<Feature> features){
		this.features = features;
	}

	static
	private List<FieldName> extractActiveFields(List<Feature> features){
		Set<FieldName> names = new LinkedHashSet<>();

		for(Feature feature : features){
			names.add(feature.getName());
		}

		List<FieldName> result = new ArrayList<>(names);

		return result;
	}
}