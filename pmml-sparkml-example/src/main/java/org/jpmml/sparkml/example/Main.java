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
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.beust.jcommander.DefaultUsageFormatter;
import com.beust.jcommander.IUsageFormatter;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.converter.NullSplitter;
import org.jpmml.model.JAXBSerializer;
import org.jpmml.model.metro.MetroJAXBSerializer;
import org.jpmml.sparkml.ArchiveUtil;
import org.jpmml.sparkml.DatasetUtil;
import org.jpmml.sparkml.PMMLBuilder;
import org.jpmml.sparkml.PipelineModelUtil;
import org.jpmml.sparkml.model.HasPredictionModelOptions;
import org.jpmml.sparkml.model.HasRegressionTableOptions;
import org.jpmml.sparkml.model.HasTreeOptions;
import org.jpmml.telemetry.Incident;
import org.jpmml.telemetry.TelemetryClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {

	@Parameter (
		names = "--pipeline-input",
		description = "Pipeline ML input ZIP file or directory",
		required = true,
		order = 1
	)
	private File pipelineInput = null;

	@Parameter (
		names = "--schema-input",
		description = "Schema JSON input file",
		required = true,
		order = 2
	)
	private File schemaInput = null;

	@Parameter (
		names = "--pmml-output",
		description = "PMML output file",
		required = true,
		order = 3
	)
	private File output = null;

	@Parameter (
		names = "--field-names",
		description = "Mapping from column name to data field name(s)",
		splitter = NullSplitter.class,
		arity = 2,
		order = 4
	)
	private List<String> fieldNames = new ArrayList<>();

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
		order = 5
	)
	private Boolean compact = Boolean.TRUE;

	/**
	 * @see HasTreeOptions#OPTION_ESTIMATE_FEATURE_IMPORTANCES
	 */
	@Parameter (
		names = "--X-estimate_featureImportances",
		arity = 1,
		order = 6
	)
	private Boolean estimateFeatureImportances = Boolean.FALSE;

	/**
	 * @see HasRegressionTableOptions#OPTION_REPRESENTATION
	 */
	@Parameter (
		names = "--X-representation",
		order = 7
	)
	private String representation = null;

	@Parameter (
		names = "--help",
		description = "Show the list of configuration options and exit",
		help = true,
		order = Integer.MAX_VALUE
	)
	private boolean help = false;


	static
	public void main(String... args) throws Exception {
		Main main = new Main();

		JCommander commander = new JCommander(main);
		commander.setProgramName(Main.class.getName());

		IUsageFormatter usageFormatter = new DefaultUsageFormatter(commander);

		try {
			commander.parse(args);
		} catch(ParameterException pe){
			StringBuilder sb = new StringBuilder();

			sb.append(pe.toString());
			sb.append("\n");

			usageFormatter.usage(sb);

			System.err.println(sb.toString());

			System.exit(-1);
		}

		if(main.help){
			StringBuilder sb = new StringBuilder();

			usageFormatter.usage(sb);

			System.out.println(sb.toString());

			System.exit(0);
		}

		try {
			main.run();
		} catch(FileNotFoundException fnfe){
			throw fnfe;
		} catch(Exception e){
			Package _package = Main.class.getPackage();

			Map<String, String> environment = new LinkedHashMap<>();
			environment.put("jpmml-sparkml", _package.getImplementationVersion());

			Incident incident = new Incident()
				.setProject("jpmml-sparkml")
				.setEnvironment(environment)
				.setException(e);

			try {
				TelemetryClient.report("https://telemetry.jpmml.org/v1/incidents", incident);
			} catch(IOException ioe){
				// Ignored
			}

			throw e;
		}
	}

	private void run() throws Exception {
		SparkSession sparkSession = SparkSession.builder()
			.getOrCreate();

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

		StructType schema;

		try {
			logger.info("Loading schema..");

			long begin = System.currentTimeMillis();
			schema = DatasetUtil.loadSchema(this.schemaInput);
			long end = System.currentTimeMillis();

			logger.info("Loaded schema in {} ms.", (end - begin));
		} catch(Exception e){
			logger.error("Failed to load schema", e);

			throw e;
		}

		PMMLBuilder pmmlBuilder = new PMMLBuilder(schema, pipelineModel);

		Map<String, Object> options = new LinkedHashMap<>();
		options.put(HasPredictionModelOptions.OPTION_KEEP_PREDICTIONCOL, this.keepPredictionCol);
		options.put(HasTreeOptions.OPTION_COMPACT, this.compact);
		options.put(HasTreeOptions.OPTION_ESTIMATE_FEATURE_IMPORTANCES, this.estimateFeatureImportances);
		options.put(HasRegressionTableOptions.OPTION_REPRESENTATION, this.representation);

		pmmlBuilder.putOptions(options);

		for(int i = 0; i < this.fieldNames.size(); i += 2){
			String column = this.fieldNames.get(i);
			List<String> names = Arrays.asList(this.fieldNames.get(i + 1).split(","));

			pmmlBuilder.putFieldNames(column, names);
		}

		PMML pmml;

		try {
			logger.info("Converting pipeline model to PMML..");

			long begin = System.currentTimeMillis();
			pmml = pmmlBuilder.build();
			long end = System.currentTimeMillis();

			logger.info("Converted pipeline to PMML in {} ms.", (end - begin));
		} catch(Exception e){
			logger.info("Failed to convert pipeline to PMML", e);

			throw e;
		}

		try(OutputStream os = new FileOutputStream(this.output)){
			logger.info("Marshalling PMML..");

			JAXBSerializer jaxbSerializer = new MetroJAXBSerializer();

			long begin = System.currentTimeMillis();
			jaxbSerializer.serializePretty(pmml, os);
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