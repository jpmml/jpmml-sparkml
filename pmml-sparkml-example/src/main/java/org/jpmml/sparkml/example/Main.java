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
package org.jpmml.sparkml.example;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.LinkedHashMap;
import java.util.Map;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.google.common.io.CharStreams;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.model.metro.MetroJAXBUtil;
import org.jpmml.sparkml.ArchiveUtil;
import org.jpmml.sparkml.PMMLBuilder;
import org.jpmml.sparkml.PipelineModelUtil;
import org.jpmml.sparkml.model.HasPredictionModelOptions;
import org.jpmml.sparkml.model.HasRegressionTableOptions;
import org.jpmml.sparkml.model.HasTreeOptions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {

	@Parameter (
		names = "--help",
		description = "Show the list of configuration options and exit",
		help = true
	)
	private boolean help = false;

	@Parameter (
		names = "--schema-input",
		description = "Schema JSON input file",
		required = true
	)
	private File schemaInput = null;

	@Parameter (
		names = "--pipeline-input",
		description = "Pipeline ML input ZIP file or directory",
		required = true
	)
	private File pipelineInput = null;

	@Parameter (
		names = "--pmml-output",
		description = "PMML output file",
		required = true
	)
	private File output = null;

	/**
	 * @see HasPredictionModelOptions#OPTION_KEEP_PREDICTIONCOL
	 */
	@Parameter (
		names = "--X-keep_predictionCol",
		arity = 1,
		hidden = true
	)
	private Boolean keepPredictionCol = Boolean.TRUE;

	/**
	 * @see HasTreeOptions#OPTION_COMPACT
	 */
	@Parameter (
		names = "--X-compact",
		arity = 1,
		hidden = true
	)
	private Boolean compact = Boolean.TRUE;

	/**
	 * @see HasTreeOptions#OPTION_ESTIMATE_FEATURE_IMPORTANCES
	 */
	@Parameter (
		names = "--X-estimate_featureImportances",
		arity = 1,
		hidden = true
	)
	private Boolean estimateFeatureImportances = Boolean.FALSE;

	/**
	 * @see HasRegressionTableOptions#OPTION_LOOKUP_THRESHOLD
	 */
	@Parameter (
		names = "--X-lookup_threshold",
		hidden = true
	)
	private Integer lookupThreshold = null;

	/**
	 * @see HasRegressionTableOptions#OPTION_REPRESENTATION
	 */
	@Parameter (
		names = "--X-representation",
		hidden = true
	)
	private String representation = null;


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
		SparkSession sparkSession = SparkSession.builder()
			.getOrCreate();

		StructType schema;

		try(InputStream is = new FileInputStream(this.schemaInput)){
			logger.info("Loading schema..");

			String json = CharStreams.toString(new InputStreamReader(is, "UTF-8"));

			long begin = System.currentTimeMillis();
			schema = (StructType)DataType.fromJson(json);
			long end = System.currentTimeMillis();

			logger.info("Loaded schema in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to load schema", e);

			throw e;
		}

		PipelineModel pipelineModel;

		try {
			logger.info("Loading pipeline model..");

			if(this.pipelineInput.isFile()){
				this.pipelineInput = ArchiveUtil.uncompress(this.pipelineInput);
			}

			long begin = System.currentTimeMillis();
			pipelineModel = PipelineModelUtil.load(sparkSession, this.pipelineInput);
			long end = System.currentTimeMillis();

			logger.info("Loaded pipeline model in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to load pipeline model", e);

			throw e;
		}

		Map<String, Object> options = new LinkedHashMap<>();
		options.put(HasPredictionModelOptions.OPTION_KEEP_PREDICTIONCOL, this.keepPredictionCol);
		options.put(HasTreeOptions.OPTION_COMPACT, this.compact);
		options.put(HasTreeOptions.OPTION_ESTIMATE_FEATURE_IMPORTANCES, this.estimateFeatureImportances);
		options.put(HasRegressionTableOptions.OPTION_LOOKUP_THRESHOLD, this.lookupThreshold);
		options.put(HasRegressionTableOptions.OPTION_REPRESENTATION, this.representation);

		PMML pmml;

		try {
			logger.info("Converting pipeline model to PMML..");

			long begin = System.currentTimeMillis();
			pmml = new PMMLBuilder(schema, pipelineModel)
				.putOptions(options)
				.build();
			long end = System.currentTimeMillis();

			logger.info("Converted pipeline to PMML in {} ms.", (end - begin));
		} catch(Exception e){
			logger.info("Failed to convert pipeline to PMML", e);

			throw e;
		}

		try(OutputStream os = new FileOutputStream(this.output)){
			logger.info("Marshalling PMML..");

			long begin = System.currentTimeMillis();
			MetroJAXBUtil.marshalPMML(pmml, os);
			long end = System.currentTimeMillis();

			logger.info("Marshalled PMML in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to marshal PMML", e);

			throw e;
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

	private static final Logger logger = LoggerFactory.getLogger(Main.class);
}