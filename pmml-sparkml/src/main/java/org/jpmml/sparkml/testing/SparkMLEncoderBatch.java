/*
 * Copyright (c) 2020 Villu Ruusmann
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
package org.jpmml.sparkml.testing;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import com.google.common.io.ByteStreams;
import com.google.common.io.CharStreams;
import com.google.common.io.MoreFiles;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.util.MLReader;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.converter.testing.ModelEncoderBatch;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.PMMLBuilder;
import org.jpmml.sparkml.ZipUtil;
import org.jpmml.sparkml.model.HasRegressionTableOptions;

abstract
public class SparkMLEncoderBatch extends ModelEncoderBatch {

	public SparkMLEncoderBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		super(algorithm, dataset, columnFilter, equivalence);
	}

	@Override
	abstract
	public SparkMLEncoderBatchTest getArchiveBatchTest();

	@Override
	public List<Map<String, Object>> getOptionsMatrix(){
		// XXX
		Map<String, Object> options = new LinkedHashMap<>();
		options.put(HasRegressionTableOptions.OPTION_LOOKUP_THRESHOLD, 5);

		return Collections.singletonList(options);
	}

	public String getSchemaJsonPath(){
		return "/schema/" + getDataset() + ".json";
	}

	public String getPipelineZipPath(){
		return "/pipeline/" + getAlgorithm() + getDataset() + ".zip";
	}

	public Dataset<Row> getVerificationDataset(StructType schema, Dataset<Row> inputDataset){
		List<StructField> fields = Arrays.asList(schema.fields());

		for(StructField field : fields){
			Column column = inputDataset.apply(field.name()).cast(field.dataType());

			inputDataset = inputDataset.withColumn("tmp_" + field.name(), column).drop(field.name()).withColumnRenamed("tmp_" + field.name(), field.name());
		}

		return inputDataset.sample(false, 0.05d, 63317);
	}

	@Override
	public PMML getPMML() throws Exception {
		SparkMLEncoderBatchTest archiveBatchTest = getArchiveBatchTest();

		SparkSession sparkSession = archiveBatchTest.getSparkSession();
		if(sparkSession == null){
			throw new IllegalStateException();
		}

		List<File> tmpResources = new ArrayList<>();

		StructType schema;

		try(InputStream is = open(getSchemaJsonPath())){
			String json = CharStreams.toString(new InputStreamReader(is, "UTF-8"));

			schema = (StructType)DataType.fromJson(json);
		}

		PipelineModel pipelineModel;

		try(InputStream is = open(getPipelineZipPath())){
			File tmpZipFile = File.createTempFile(getAlgorithm() + getDataset(), ".zip");

			tmpResources.add(tmpZipFile);

			try(OutputStream os = new FileOutputStream(tmpZipFile)){
				ByteStreams.copy(is, os);
			}

			File tmpPipelineDir = File.createTempFile(getAlgorithm() + getDataset(), "");
			if(!tmpPipelineDir.delete()){
				throw new IOException();
			}

			tmpResources.add(tmpPipelineDir);

			ZipUtil.uncompress(tmpZipFile, tmpPipelineDir);

			MLReader<PipelineModel> mlReader = new PipelineModel.PipelineModelReader();
			mlReader.session(sparkSession);

			pipelineModel = mlReader.load(tmpPipelineDir.getAbsolutePath());
		}

		Dataset<Row> inputDataset;

		try(InputStream is = open(getInputCsvPath())){
			File tmpCsvFile = File.createTempFile(getDataset(), ".csv");

			tmpResources.add(tmpCsvFile);

			try(OutputStream os = new FileOutputStream(tmpCsvFile)){
				ByteStreams.copy(is, os);
			}

			inputDataset = sparkSession.read()
				.format("csv")
				.option("header", true)
				.option("inferSchema", false)
				.load(tmpCsvFile.getAbsolutePath());
		}

		Map<String, Object> options = getOptions();

		PMMLBuilder pmmlBuilder = new PMMLBuilder(schema, pipelineModel)
			.putOptions(options);

		Dataset<Row> verificationDataset = getVerificationDataset(schema, inputDataset);
		if(verificationDataset != null){
			Equivalence<?> equivalence = getEquivalence();

			double precision = 1e-14;
			double zeroThreshold = 1e-14;

			if(equivalence instanceof PMMLEquivalence){
				PMMLEquivalence pmmlEquivalence = (PMMLEquivalence)equivalence;

				precision = pmmlEquivalence.getPrecision();
				zeroThreshold = pmmlEquivalence.getZeroThreshold();
			}

			pmmlBuilder = pmmlBuilder.verify(verificationDataset, precision, zeroThreshold);
		}

		PMML pmml = pmmlBuilder.build();

		validatePMML(pmml);

		for(File tmpResource : tmpResources){
			MoreFiles.deleteRecursively(tmpResource.toPath());
		}

		return pmml;
	}
}