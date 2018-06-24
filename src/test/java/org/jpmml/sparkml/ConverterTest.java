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

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.io.ByteStreams;
import com.google.common.io.CharStreams;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.shared.HasFitIntercept;
import org.apache.spark.ml.util.MLReader;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Visitor;
import org.jpmml.converter.visitors.CellTransformer;
import org.jpmml.evaluator.ArchiveBatch;
import org.jpmml.evaluator.IntegrationTest;
import org.jpmml.evaluator.IntegrationTestBatch;
import org.jpmml.evaluator.PMMLEquivalence;
import org.jpmml.sparkml.model.HasRegressionOptions;
import org.junit.AfterClass;
import org.junit.BeforeClass;

abstract
public class ConverterTest extends IntegrationTest {

	public ConverterTest(){
		super(new PMMLEquivalence(1e-14, 1e-14));
	}

	@Override
	protected ArchiveBatch createBatch(String name, String dataset, Predicate<FieldName> predicate){
		Predicate<FieldName> excludePredictionFields = excludeFields(FieldName.create("prediction"), FieldName.create("pmml(prediction)"));

		if(predicate == null){
			predicate = excludePredictionFields;
		} else

		{
			predicate = Predicates.and(predicate, excludePredictionFields);
		}

		ArchiveBatch result = new IntegrationTestBatch(name, dataset, predicate){

			@Override
			public IntegrationTest getIntegrationTest(){
				return ConverterTest.this;
			}

			@Override
			public PMML getPMML() throws Exception {
				StructType schema;

				try(InputStream is = open("/schema/" + getDataset() + ".json")){
					String json = CharStreams.toString(new InputStreamReader(is, "UTF-8"));

					schema = (StructType)DataType.fromJson(json);
				}

				PipelineModel pipelineModel;

				try(InputStream is = open("/pipeline/" + getName() + getDataset() + ".zip")){
					File tmpZipFile = File.createTempFile(getName() + getDataset(), ".zip");

					try(OutputStream os = new FileOutputStream(tmpZipFile)){
						ByteStreams.copy(is, os);
					}

					File tmpDir = File.createTempFile(getName() + getDataset(), "");
					if(!tmpDir.delete()){
						throw new IOException();
					}

					ZipUtil.uncompress(tmpZipFile, tmpDir);

					MLReader<PipelineModel> mlReader = new PipelineModel.PipelineModelReader();
					mlReader.session(ConverterTest.sparkSession);

					pipelineModel = mlReader.load(tmpDir.getAbsolutePath());
				}

				PMMLBuilder pmmlBuilder = new PMMLBuilder(schema, pipelineModel);

				Transformer[] transformers = pipelineModel.stages();
				for(Transformer transformer : transformers){

					if(transformer instanceof HasFitIntercept){
						pmmlBuilder.putOption(transformer, HasRegressionOptions.OPTION_LOOKUP_THRESHOLD, 3);
					}
				}

				PMML pmml = pmmlBuilder.build();

				Visitor visitor = new CellTransformer();
				visitor.applyTo(pmml);

				ensureValidity(pmml);

				return pmml;
			}
		};

		return result;
	}

	@BeforeClass
	static
	public void createSparkSession(){
		SparkSession.Builder builder = SparkSession.builder()
			.appName("test")
			.master("local[1]")
			.config("spark.ui.enabled", false);

		SparkSession sparkSession = builder.getOrCreate();

		SparkContext sparkContext = sparkSession.sparkContext();
		sparkContext.setLogLevel("ERROR");

		ConverterTest.sparkSession = sparkSession;
	}

	@AfterClass
	static
	public void destroySparkSession(){
		ConverterTest.sparkSession.stop();
		ConverterTest.sparkSession = null;
	}

	public static SparkSession sparkSession = null;
}