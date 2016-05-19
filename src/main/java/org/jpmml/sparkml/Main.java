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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.model.MetroJAXBUtil;
import org.jpmml.model.SerializationUtil;

public class Main {

	@Parameter (
		names = "--help",
		description = "Show the list of configuration options and exit",
		help = true
	)
	private boolean help = false;

	@Parameter (
		names = "--ser-schema-input",
		description = "Schema SER input file",
		required = true
	)
	private File schemaInput = null;

	@Parameter (
		names = "--ser-pipeline-input",
		description = "Pipeline SER input file",
		required = true
	)
	private File pipelineInput = null;

	@Parameter (
		names = "--pmml-output",
		description = "PMML output file",
		required = true
	)
	private File output = null;


	static
	public void main(String... args) throws Exception {
		Main main = new Main();

		JCommander commander = new JCommander(main);
		commander.setProgramName(Main.class.getName());

		try {
			commander.parse(args);
		} catch(ParameterException pe){
			StringBuilder sb = new StringBuilder();

			sb.append(pe.toString());
			sb.append("\n");

			commander.usage(sb);

			System.err.println(sb.toString());

			System.exit(-1);
		}

		if(main.help){
			StringBuilder sb = new StringBuilder();

			commander.usage(sb);

			System.out.println(sb.toString());

			System.exit(0);
		}

		main.run();
	}

	private void run() throws Exception {
		StructType schema = (StructType)deserialize(this.schemaInput);

		PipelineModel pipelineModel = (PipelineModel)deserialize(this.pipelineInput);

		PMML pmml = ConverterUtil.toPMML(schema, pipelineModel);

		try(OutputStream os = new FileOutputStream(this.output)){
			MetroJAXBUtil.marshalPMML(pmml, os);
		}
	}

	public File getSchemaInput(){
		return this.schemaInput;
	}

	public void setSchemaInput(File schemaInput){
		this.schemaInput = schemaInput;
	}

	public File getPipelineInput(){
		return this.pipelineInput;
	}

	public void setPipelineInput(File pipelineInput){
		this.pipelineInput = pipelineInput;
	}

	public File getOutput(){
		return this.output;
	}

	public void setOutput(File output){
		this.output = output;
	}

	static
	private Object deserialize(File file) throws ClassNotFoundException, IOException {

		try(InputStream is = new FileInputStream(file)){
			return SerializationUtil.deserialize(is);
		}
	}
}