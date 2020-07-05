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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
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
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.ArchiveBatch;
import org.jpmml.evaluator.testing.IntegrationTest;
import org.jpmml.evaluator.testing.IntegrationTestBatch;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.model.HasRegressionTableOptions;
import org.junit.AfterClass;
import org.junit.BeforeClass;

abstract
public class SparkMLTest extends IntegrationTest {

	public SparkMLTest(){
		super(new PMMLEquivalence(1e-14, 1e-14));
	}

	public Map<String, Object> getOptions(String name, String dataset){
		Map<String, Object> options = new LinkedHashMap<>();
		options.put(HasRegressionTableOptions.OPTION_LOOKUP_THRESHOLD, 3);

		return options;
	}

	@Override
	protected ArchiveBatch createBatch(String name, String dataset, Predicate<ResultField> predicate, Equivalence<Object> equivalence){
		Predicate<ResultField> excludePredictionFields = excludeFields(FieldName.create("prediction"), FieldName.create("pmml(prediction)"));

		predicate = predicate.and(excludePredictionFields);

		ArchiveBatch result = new IntegrationTestBatch(name, dataset, predicate, equivalence){

			@Override
			public IntegrationTest getIntegrationTest(){
				return SparkMLTest.this;
			}

			@Override
			public PMML getPMML() throws Exception {
				List<File> tmpResources = new ArrayList<>();

				StructType schema;

				try(InputStream is = open("/schema/" + getDataset() + ".json")){
					String json = CharStreams.toString(new InputStreamReader(is, "UTF-8"));

					schema = (StructType)DataType.fromJson(json);
				}

				PipelineModel pipelineModel;

				try(InputStream is = open("/pipeline/" + getName() + getDataset() + ".zip")){
					File tmpZipFile = File.createTempFile(getName() + getDataset(), ".zip");

					tmpResources.add(tmpZipFile);

					try(OutputStream os = new FileOutputStream(tmpZipFile)){
						ByteStreams.copy(is, os);
					}

					File tmpPipelineDir = File.createTempFile(getName() + getDataset(), "");
					if(!tmpPipelineDir.delete()){
						throw new IOException();
					}

					tmpResources.add(tmpPipelineDir);

					ZipUtil.uncompress(tmpZipFile, tmpPipelineDir);

					MLReader<PipelineModel> mlReader = new PipelineModel.PipelineModelReader();
					mlReader.session(SparkMLTest.sparkSession);

					pipelineModel = mlReader.load(tmpPipelineDir.getAbsolutePath());
				}

				Dataset<Row> dataset;

				try(InputStream is = open("/csv/" + getDataset() + ".csv")){
					File tmpCsvFile = File.createTempFile(getDataset(), ".csv");

					tmpResources.add(tmpCsvFile);

					try(OutputStream os = new FileOutputStream(tmpCsvFile)){
						ByteStreams.copy(is, os);
					}

					dataset = SparkMLTest.sparkSession.read()
						.format("csv")
						.option("header", true)
						.option("inferSchema", false)
						.load(tmpCsvFile.getAbsolutePath());

					StructField[] fields = schema.fields();
					for(StructField field : fields){
						Column column = dataset.apply(field.name()).cast(field.dataType());

						dataset = dataset.withColumn("tmp_" + field.name(), column).drop(field.name()).withColumnRenamed("tmp_" + field.name(), field.name());
					}

					dataset = dataset.sample(false, 0.05d, 63317);
				}

				Map<String, Object> options = getOptions(getName(), getDataset());

				double precision = 1e-14;
				double zeroThreshold = 1e-14;

				if(equivalence instanceof PMMLEquivalence){
					PMMLEquivalence pmmlEquivalence = (PMMLEquivalence)equivalence;

					precision = pmmlEquivalence.getPrecision();
					zeroThreshold = pmmlEquivalence.getZeroThreshold();
				}

				PMMLBuilder pmmlBuilder = new PMMLBuilder(schema, pipelineModel)
					.putOptions(options)
					.verify(dataset, precision, zeroThreshold);

				PMML pmml = pmmlBuilder.build();

				validatePMML(pmml);

				for(File tmpResource : tmpResources){
					MoreFiles.deleteRecursively(tmpResource.toPath());
				}

				return pmml;
			}
		};

		return result;
	}

	@BeforeClass
	static
	public void createSparkSession(){
		SparkMLTest.sparkSession = SparkSessionUtil.createSparkSession();
	}

	@AfterClass
	static
	public void destroySparkSession(){
		SparkMLTest.sparkSession = SparkSessionUtil.destroySparkSession(SparkMLTest.sparkSession);
	}

	public static SparkSession sparkSession = null;
}