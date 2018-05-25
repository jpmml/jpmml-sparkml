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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.IntegralType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.WildcardFeature;

public class SparkMLEncoder extends ModelEncoder {

	private StructType schema = null;

	private Map<String, List<Feature>> columnFeatures = new LinkedHashMap<>();


	public SparkMLEncoder(StructType schema){
		this.schema = schema;
	}

	public boolean hasFeatures(String column){
		return this.columnFeatures.containsKey(column);
	}

	public Feature getOnlyFeature(String column){
		List<Feature> features = getFeatures(column);

		return Iterables.getOnlyElement(features);
	}

	public List<Feature> getFeatures(String column){
		List<Feature> features = this.columnFeatures.get(column);

		if(features == null){
			FieldName name = FieldName.create(column);

			DataField dataField = getDataField(name);
			if(dataField == null){
				dataField = createDataField(name);
			}

			Feature feature;

			DataType dataType = dataField.getDataType();
			switch(dataType){
				case STRING:
					feature = new WildcardFeature(this, dataField);
					break;
				case INTEGER:
				case DOUBLE:
					feature = new ContinuousFeature(this, dataField);
					break;
				case BOOLEAN:
					feature = new BooleanFeature(this, dataField);
					break;
				default:
					throw new IllegalArgumentException("Data type " + dataType + " is not supported");
			}

			return Collections.singletonList(feature);
		}

		return features;
	}

	public List<Feature> getFeatures(String column, int[] indices){
		List<Feature> features = getFeatures(column);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < indices.length; i++){
			int index = indices[i];

			Feature feature = features.get(index);

			result.add(feature);
		}

		return result;
	}

	public void putOnlyFeature(String column, Feature feature){
		putFeatures(column, Collections.singletonList(feature));
	}

	public void putFeatures(String column, List<Feature> features){
		List<Feature> existingFeatures = this.columnFeatures.get(column);

		if(existingFeatures != null && existingFeatures.size() > 0){

			if(features.size() != existingFeatures.size()){
				throw new IllegalArgumentException("Expected " + existingFeatures.size() + " features, got " + features.size() + " features");
			}

			for(int i = 0; i < existingFeatures.size(); i++){
				Feature existingFeature = existingFeatures.get(i);
				Feature feature = features.get(i);

				if(!(feature.getName()).equals(existingFeature.getName())){
					throw new IllegalArgumentException();
				}
			}
		}

		this.columnFeatures.put(column, features);
	}

	public DataField createDataField(FieldName name){
		StructField field = this.schema.apply(name.getValue());

		org.apache.spark.sql.types.DataType sparkDataType = field.dataType();

		if(sparkDataType instanceof StringType){
			return createDataField(name, OpType.CATEGORICAL, DataType.STRING);
		} else

		if(sparkDataType instanceof IntegralType){
			return createDataField(name, OpType.CONTINUOUS, DataType.INTEGER);
		} else

		if(sparkDataType instanceof DoubleType){
			return createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
		} else

		if(sparkDataType instanceof BooleanType){
			return createDataField(name, OpType.CATEGORICAL, DataType.BOOLEAN);
		} else

		{
			throw new IllegalArgumentException("Expected string, integral, double or boolean type, got " + sparkDataType.typeName() + " type");
		}
	}

	public void removeDataField(FieldName name){
		Map<FieldName, DataField> dataFields = getDataFields();

		DataField dataField = dataFields.remove(name);
		if(dataField == null){
			throw new IllegalArgumentException(name.getValue());
		}
	}

	public void removeDerivedField(FieldName name){
		Map<FieldName, DerivedField> derivedFields = getDerivedFields();

		DerivedField derivedField = derivedFields.remove(name);
		if(derivedField == null){
			throw new IllegalArgumentException(name.getValue());
		}
	}

	@Override
	public Map<FieldName, DataField> getDataFields(){
		return super.getDataFields();
	}

	@Override
	public Map<FieldName, DerivedField> getDerivedFields(){
		return super.getDerivedFields();
	}
}