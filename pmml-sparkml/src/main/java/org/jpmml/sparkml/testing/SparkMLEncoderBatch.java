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
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import com.google.common.io.ByteStreams;
import com.google.common.io.MoreFiles;
import com.google.common.io.RecursiveDeleteOption;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.converter.testing.ModelEncoderBatch;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.ArchiveUtil;
import org.jpmml.sparkml.DatasetUtil;
import org.jpmml.sparkml.PMMLBuilder;
import org.jpmml.sparkml.PipelineModelUtil;

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
		Map<String, Object> options = new LinkedHashMap<>();

		return Collections.singletonList(options);
	}

	public String getPipelineModelZipPath(){
		return "/pipeline/" + getAlgorithm() + getDataset() + ".zip";
	}

	public String getSchemaJsonPath(){
		return "/schema/" + getDataset() + ".json";
	}

	public Dataset<Row> getVerificationDataset(Dataset<Row> inputDataset){
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

		StructType schema = loadSchema(sparkSession, tmpResources);

		PipelineModel pipelineModel = loadPipelineModel(sparkSession, tmpResources);

		schema = updateSchema(schema, pipelineModel);

		Dataset<Row> inputDataset = loadInput(sparkSession, tmpResources);

		inputDataset = DatasetUtil.castColumns(inputDataset, schema);

		Map<String, Object> options = getOptions();

		PMMLBuilder pmmlBuilder = new PMMLBuilder(schema, pipelineModel)
			.putOptions(options);

		Dataset<Row> verificationDataset = getVerificationDataset(inputDataset);
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
			MoreFiles.deleteRecursively(tmpResource.toPath(), RecursiveDeleteOption.ALLOW_INSECURE);
		}

		return pmml;
	}

	protected StructType loadSchema(SparkSession sparkSession, List<File> tmpResources) throws IOException {

		try(InputStream is = open(getSchemaJsonPath())){
			File tmpSchemaFile = toTmpFile(is, getDataset(), ".json");

			tmpResources.add(tmpSchemaFile);

			return DatasetUtil.loadSchema(tmpSchemaFile);
		}
	}

	protected PipelineModel loadPipelineModel(SparkSession sparkSession, List<File> tmpResources) throws IOException {

		try(InputStream is = open(getPipelineModelZipPath())){
			File tmpZipFile = toTmpFile(is, getAlgorithm() + getDataset(), ".zip");

			tmpResources.add(tmpZipFile);

			File tmpPipelineModelDir = ArchiveUtil.uncompress(tmpZipFile);

			tmpResources.add(tmpPipelineModelDir);

			return PipelineModelUtil.load(sparkSession, tmpPipelineModelDir);
		}
	}

	protected StructType updateSchema(StructType schema, PipelineModel pipelineModel){
		return schema;
	}

	protected Dataset<Row> loadInput(SparkSession sparkSession, List<File> tmpResources) throws IOException {

		try(InputStream is = open(getInputCsvPath())){
			File tmpCsvFile = toTmpFile(is, getDataset(), ".csv");

			tmpResources.add(tmpCsvFile);

			return DatasetUtil.loadCsv(sparkSession, tmpCsvFile);
		}
	}

	static
	protected File toTmpFile(InputStream is, String prefix, String suffix) throws IOException {
		File tmpFile = File.createTempFile(prefix, suffix);

		try(OutputStream os = new FileOutputStream(tmpFile)){
			ByteStreams.copy(is, os);
		}

		return tmpFile;
	}
}