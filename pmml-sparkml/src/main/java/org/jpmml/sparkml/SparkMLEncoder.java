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
import java.util.Objects;

import com.google.common.collect.Iterables;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Field;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.association.Item;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.ModelEncoder;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.TypeUtil;
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
			String name = column;

			DataField dataField = getDataField(name);
			if(dataField == null){
				dataField = createDataField(name);
			}

			Feature feature = createFeature(dataField);

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

	public DataField createDataField(String name){
		StructType schema = getSchema();

		StructField field = schema.apply(name);

		DataType dataType = DatasetUtil.translateDataType(field.dataType());

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
}