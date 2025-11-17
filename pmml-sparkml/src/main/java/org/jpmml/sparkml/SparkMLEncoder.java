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
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Field;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Value;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.association.Item;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.FieldUtil;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.model.visitors.AbstractVisitor;

public class SparkMLEncoder extends ModelEncoder {

	private StructType schema = null;

	private ConverterFactory converterFactory = null;

	private Map<String, List<Feature>> columnFeatures = new LinkedHashMap<>();


	public SparkMLEncoder(StructType schema, ConverterFactory converterFactory){
		setSchema(schema);
		setConverterFactory(converterFactory);
	}

	@Override
	public PMML encodePMML(Model model){
		PMML pmml = super.encodePMML(model);

		Visitor visitor = new AbstractVisitor(){

			@Override
			public VisitorAction visit(Item item){
				item.setField((String)null);

				return super.visit(item);
			}
		};
		visitor.applyTo(pmml);

		return pmml;
	}

	public Field<?> toContinuous(Feature feature){
		return toContinuous(feature.getName());
	}

	public Field<?> toCategorical(Feature feature, List<?> values){

		values:
		if(feature instanceof ObjectFeature){

			if(values == null || values.isEmpty()){
				break values;
			}

			Field<?> field = feature.getField();

			if(field instanceof DataField){
				DataField dataField = (DataField)field;

				List<?> existingValues = FieldUtil.getValues(dataField);
				if(existingValues != null && !existingValues.isEmpty()){
					DataType dataType = dataField.requireDataType();

					if((existingValues.size() == values.size())  && (parseValues(dataType, existingValues)).equals(parseValues(dataType, values))){
						FieldUtil.clearValues(dataField, Value.Property.VALID);
					}
				}
			}
		}

		return toCategorical(feature.getName(), values);
	}

	public Field<?> toOrdinal(Feature feature, List<?> values){
		return toOrdinal(feature.getName(), values);
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
			StructType schema = getSchema();

			StructField field = schema.apply(column);

			org.apache.spark.sql.types.DataType sparkDataType = field.dataType();

			if(sparkDataType instanceof VectorUDT){
				Metadata metadata = field.metadata();

				int numFeatures = (int)metadata.getLong("numFeatures");
				if(numFeatures < 0){
					throw new IllegalArgumentException();
				}

				List<String> fieldNames = getFieldNames(column);
				if(fieldNames != null && fieldNames.size() != numFeatures){
					throw new IllegalArgumentException("Expected " + numFeatures + " data field names, got " + fieldNames.size()  + " data field names");
				}

				List<Feature> result = new ArrayList<>();

				for(int i = 0; i < numFeatures; i++){
					String name;

					if(fieldNames != null){
						name = fieldNames.get(i);
					} else

					{
						name = FieldNameUtil.select(column, i);
					}

					DataField dataField = getDataField(name);
					if(dataField == null){
						dataField = createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
					}

					result.add(createFeature(dataField));
				}

				return result;
			} else

			{
				List<String> fieldNames = getFieldNames(column);
				if(fieldNames != null && fieldNames.size() != 1){
					throw new IllegalArgumentException("Expected 1 data field name, got " + fieldNames.size() + " data field names");
				}

				String name;

				if(fieldNames != null){
					name = Iterables.getOnlyElement(fieldNames);
				} else

				{
					name = column;
				}

				DataField dataField = getDataField(name);
				if(dataField == null){
					dataField = createDataField(column, name);
				}

				Feature feature = createFeature(dataField);

				return Collections.singletonList(feature);
			}
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

		if(existingFeatures != null && !existingFeatures.isEmpty()){
			SchemaUtil.checkSize(existingFeatures.size(), features);

			for(int i = 0; i < existingFeatures.size(); i++){
				Feature existingFeature = existingFeatures.get(i);
				Feature feature = features.get(i);

				if(!(feature.getName()).equals(existingFeature.getName())){
					throw new IllegalArgumentException("Expected feature column '" + existingFeature.getName() + "', got feature column '" + feature.getName() + "'");
				}
			}
		}

		this.columnFeatures.put(column, features);
	}

	public List<String> getFieldNames(String column){
		return null;
	}

	public DataField createDataField(String column, String name){
		StructType schema = getSchema();

		StructField field = schema.apply(column);

		org.apache.spark.sql.types.DataType sparkDataType = field.dataType();

		DataType dataType = DatasetUtil.translateDataType(sparkDataType);

		OpType opType = TypeUtil.getOpType(dataType);

		return createDataField(name, opType, dataType);
	}

	public Feature createFeature(Field<?> field){
		DataType dataType = field.requireDataType();
		OpType opType = field.requireOpType();

		switch(dataType){
			case STRING:
			case INTEGER:
			case FLOAT:
			case DOUBLE:
			case BOOLEAN:
				return FeatureUtil.createFeature(field, this);
			default:
				throw new IllegalArgumentException("Data type " + dataType + " is not supported");
		}
	}

	public StructType getSchema(){
		return this.schema;
	}

	private void setSchema(StructType schema){
		this.schema = Objects.requireNonNull(schema);
	}

	public ConverterFactory getConverterFactory(){
		return this.converterFactory;
	}

	private void setConverterFactory(ConverterFactory converterFactory){
		this.converterFactory = Objects.requireNonNull(converterFactory);
	}

	static
	private Set<?> parseValues(DataType dataType, Collection<?> values){
		return values.stream()
			.map(value -> parseValue(dataType, value))
			.collect(Collectors.toSet());
	}

	static
	private Object parseValue(DataType dataType, Object value){
		String string = ValueUtil.asString(value);

		switch(dataType){
			case STRING:
				return string;
			case INTEGER:
				return Long.valueOf(string);
			case FLOAT:
			case DOUBLE:
				return Double.valueOf(string);
			case BOOLEAN:
				return Boolean.valueOf(string);
			default:
				throw new IllegalArgumentException("Data type " + dataType + " is not supported");
		}
	}
}