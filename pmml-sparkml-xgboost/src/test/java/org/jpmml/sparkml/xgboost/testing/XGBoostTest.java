/*
 * Copyright (c) 2017 Villu Ruusmann
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
package org.jpmml.sparkml.xgboost.testing;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel;
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.util.MLReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.VerificationField;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.converter.testing.OptionsUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sparkml.ArchiveUtil;
import org.jpmml.sparkml.PipelineModelUtil;
import org.jpmml.sparkml.testing.SparkMLEncoderBatch;
import org.jpmml.sparkml.testing.SparkMLEncoderBatchTest;
import org.jpmml.sparkml.xgboost.HasSparkMLXGBoostOptions;
import org.jpmml.xgboost.HasXGBoostOptions;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

public class XGBoostTest extends SparkMLEncoderBatchTest implements Datasets {

	public XGBoostTest(){
		super(new FloatEquivalence(12));
	}

	@Override
	public SparkMLEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		columnFilter = columnFilter.and(excludePredictionFields());

		SparkMLEncoderBatch result = new SparkMLEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public XGBoostTest getArchiveBatchTest(){
				return XGBoostTest.this;
			}

			@Override
			public List<Map<String, Object>> getOptionsMatrix(){
				Map<String, Object> options = new LinkedHashMap<>();

				options.put(HasSparkMLXGBoostOptions.OPTION_INPUT_FLOAT, new Boolean[]{false, true});

				options.put(HasXGBoostOptions.OPTION_COMPACT, new Boolean[]{false, true});
				options.put(HasXGBoostOptions.OPTION_PRUNE, false);

				return OptionsUtil.generateOptionsMatrix(options);
			}

			@Override
			public PMML getPMML() throws Exception {
				PMML pmml = super.getPMML();

				String dataset = getDataset();

				Visitor visitor = new AbstractVisitor(){

					@Override
					public VisitorAction visit(Model model){

						if(Objects.equals(dataset, AUDIT)){
							model.setModelVerification(null);
						}

						return super.visit(model);
					}

					@Override
					public VisitorAction visit(VerificationField verificationField){
						verificationField
							.setPrecision(1e-6d)
							.setZeroThreshold(1e-6d);

						return super.visit(verificationField);
					}
				};
				visitor.applyTo(pmml);

				return pmml;
			}

			@Override
			protected PipelineModel loadPipelineModel(SparkSession sparkSession, List<File> tmpResources) throws IOException {
				String dataset = getDataset();

				if(Objects.equals(dataset, AUDIT_NA)){
					return loadPipelineModel(sparkSession, "Transformers", "XGBoostClassificationModel", tmpResources);
				} else

				if(Objects.equals(dataset, AUTO_NA)){
					return loadPipelineModel(sparkSession, "Transformers", "XGBoostRegressionModel", tmpResources);
				} else

				{
					return super.loadPipelineModel(sparkSession, tmpResources);
				}
			}

			private PipelineModel loadPipelineModel(SparkSession sparkSession, String pipelineModelName, String modelName, List<File> tmpResources) throws IOException {
				String dataset = getDataset();

				PipelineModel pipelineModel;

				try(InputStream is = open("/pipeline/" + pipelineModelName + dataset + ".zip")){
					File tmpZipFile = toTmpFile(is, pipelineModelName + dataset, ".zip");

					tmpResources.add(tmpZipFile);

					File tmpPipelineModelDir = ArchiveUtil.uncompress(tmpZipFile);

					tmpResources.add(tmpPipelineModelDir);

					pipelineModel = PipelineModelUtil.load(sparkSession, tmpPipelineModelDir);
				}

				PredictionModel<?, ?> model;

				try(InputStream is = open("/pipeline/" + modelName + dataset + ".zip")){
					File tmpZipFile = toTmpFile(is, modelName + dataset, ".zip");

					tmpResources.add(tmpZipFile);

					File tmpModelDir = ArchiveUtil.uncompress(tmpZipFile);

					tmpResources.add(tmpModelDir);

					MLReader<?> mlReader;

					if(modelName.endsWith("ClassificationModel")){
						mlReader = new XGBoostClassificationModel.XGBoostClassificationModelReader();
					} else

					if(modelName.endsWith("RegressionModel")){
						mlReader = new XGBoostRegressionModel.XGBoostRegressionModelReader();
					} else

					{
						throw new IllegalArgumentException();
					}

					mlReader.session(sparkSession);

					model = (PredictionModel<?, ?>)mlReader.load(tmpModelDir.getAbsolutePath());
				}

				PipelineModelUtil.addStage(pipelineModel, (pipelineModel.stages()).length, model);

				return pipelineModel;
			}

			@Override
			public Dataset<Row> getVerificationDataset(Dataset<Row> inputDataset){
				String dataset = getDataset();

				if(Objects.equals(dataset, AUDIT_NA) || Objects.equals(dataset, AUTO_NA)){
					return null;
				}

				return super.getVerificationDataset(inputDataset);
			}
		};

		return result;
	}

	@Test
	public void evaluateAudit() throws Exception {
		evaluate("XGBoost", AUDIT, excludeFields(Fields.AUDIT_PROBABILITY_FALSE), new FloatEquivalence(64 + 8));
	}

	@Test
	public void evaluateAuditNA() throws Exception {
		evaluate("XGBoost", AUDIT_NA, excludeFields(Fields.AUDIT_PROBABILITY_FALSE), new FloatEquivalence(64 + 8));
	}

	@Test
	public void evaluateAuto() throws Exception {
		evaluate("XGBoost", AUTO);
	}

	@Test
	public void evaluateAutoNA() throws Exception {
		evaluate("XGBoost", AUTO_NA);
	}

	@Test
	public void evaluateHousing() throws Exception {
		evaluate("XGBoost", HOUSING);
	}

	@Test
	public void evaluateIris() throws Exception {
		evaluate("XGBoost", IRIS, new FloatEquivalence(16));
	}

	@BeforeClass
	static
	public void createSparkSession(){
		SparkMLEncoderBatchTest.createSparkSession();
	}

	@AfterClass
	static
	public void destroySparkSession(){
		SparkMLEncoderBatchTest.destroySparkSession();
	}
}