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
package org.jpmml.sparkml;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.xml.bind.JAXBException;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.model.MetroJAXBUtil;

public class PMMLBuilder {

	private StructType schema = null;

	private PipelineModel pipelineModel = null;

	private Map<String, Map<String, Object>> options = new LinkedHashMap<>();


	public PMMLBuilder(StructType schema, PipelineModel pipelineModel){
		setSchema(schema);
		setPipelineModel(pipelineModel);
	}

	public PMML build(){
		StructType schema = getSchema();
		PipelineModel pipelineModel = getPipelineModel();
		Map<String, ? extends Map<String, ?>> options = getOptions();

		return ConverterUtil.toPMML(schema, pipelineModel, options);
	}

	public byte[] buildByteArray(){
		return buildByteArray(1024 * 1024);
	}

	private byte[] buildByteArray(int size){
		PMML pmml = build();

		ByteArrayOutputStream os = new ByteArrayOutputStream(size);

		try {
			MetroJAXBUtil.marshalPMML(pmml, os);
		} catch(JAXBException je){
			throw new RuntimeException(je);
		}

		return os.toByteArray();
	}

	public File buildFile(File file) throws IOException {
		PMML pmml = build();

		OutputStream os = new FileOutputStream(file);

		try {
			MetroJAXBUtil.marshalPMML(pmml, os);
		} catch(JAXBException je){
			throw new RuntimeException(je);
		} finally {
			os.close();
		}

		return file;
	}

	public PMMLBuilder putOption(PipelineStage pipelineStage, String key, Object value){
		return putOption(pipelineStage.uid(), key, value);
	}

	public PMMLBuilder putOptions(PipelineStage pipelineStage, Map<String, ?> map){
		return putOptions(pipelineStage.uid(), map);
	}

	public PMMLBuilder putOption(String uid, String key, Object value){
		return putOptions(uid, Collections.singletonMap(key, value));
	}

	public PMMLBuilder putOptions(String uid, Map<String, ?> map){
		Map<String, Map<String, Object>> options = getOptions();

		Map<String, Object> pipelineStageOptions = options.get(uid);
		if(pipelineStageOptions == null){
			pipelineStageOptions = new LinkedHashMap<>();

			options.put(uid, pipelineStageOptions);
		}

		pipelineStageOptions.putAll(map);

		return this;
	}

	public PMMLBuilder removeOptions(PipelineStage pipelineStage){
		return removeOptions(pipelineStage.uid());
	}

	public PMMLBuilder removeOptions(String uid){
		Map<String, Map<String, Object>> options = getOptions();

		options.remove(uid);

		return this;
	}

	public StructType getSchema(){
		return this.schema;
	}

	public PMMLBuilder setSchema(StructType schema){

		if(schema == null){
			throw new IllegalArgumentException();
		}

		this.schema = schema;

		return this;
	}

	public PipelineModel getPipelineModel(){
		return this.pipelineModel;
	}

	public PMMLBuilder setPipelineModel(PipelineModel pipelineModel){

		if(pipelineModel == null){
			throw new IllegalArgumentException();
		}

		this.pipelineModel = pipelineModel;

		return this;
	}

	public Map<String, Map<String, Object>> getOptions(){
		return this.options;
	}
}