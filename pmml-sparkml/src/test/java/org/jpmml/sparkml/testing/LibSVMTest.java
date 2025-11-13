/*
 * Copyright (c) 2025 Villu Ruusmann
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
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.Table;
import org.jpmml.evaluator.TableCollector;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.DatasetUtil;
import org.jpmml.sparkml.PMMLBuilder;
import org.junit.jupiter.api.Test;

public class LibSVMTest extends SimpleSparkMLEncoderBatchTest implements SparkMLAlgorithms, Datasets {

	@Override
	public SparkMLEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		columnFilter = columnFilter.and(excludePredictionFields());

		SparkMLEncoderBatch result = new SparkMLEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public LibSVMTest getArchiveBatchTest(){
				return LibSVMTest.this;
			}

			@Override
			public PMMLBuilder createPMMLBuilder(SparkSession sparkSession, List<File> tmpResources) throws Exception {
				PMMLBuilder pmmlBuilder = super.createPMMLBuilder(sparkSession, tmpResources);

				String algorithm = getAlgorithm();
				String dataset = getDataset();

				if((LOGISTIC_REGRESSION).equals(algorithm) && (IRIS + "Vec").equals(dataset)){
					List<String> fieldNames = Arrays.asList("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width");

					pmmlBuilder = pmmlBuilder
						.putFieldNames("features", fieldNames);
				}

				return pmmlBuilder;
			}

			@Override
			public Table getInput() throws IOException {
				Table table = super.getInput();

				String algorithm = getAlgorithm();
				String dataset = getDataset();

				if((LINEAR_REGRESION).equals(algorithm) && (HOUSING + "Vec").equals(dataset)){
					table = toLibSVM(table);
				}

				return table;
			}

			@Override
			public String getInputCsvPath(){
				String path = super.getInputCsvPath();

				path = path.replace("Vec", "");

				return path;
			}

			@Override
			protected Dataset<Row> loadInputDataset(SparkSession sparkSession, List<File> tmpResources) throws IOException {
				Dataset<Row> dataset = super.loadInputDataset(sparkSession, tmpResources);

				return DatasetUtil.toLibSVM(dataset);
			}

			@Override
			protected Dataset<Row> loadVerificationDataset(SparkSession sparkSession, List<File> tmpResources){
				return null;
			}
		};

		return result;
	}

	@Test
	public void evaluateLinearRegression() throws Exception {
		evaluate(LINEAR_REGRESION, HOUSING + "Vec");
	}

	@Test
	public void evaluateLogisticRegressionIris() throws Exception {
		evaluate(LOGISTIC_REGRESSION, IRIS + "Vec", new PMMLEquivalence(1e-12, 1e-12));
	}

	static
	public Table toLibSVM(Table table){
		Function<org.jpmml.evaluator.Table.Row, Map<String, Object>> rowMapper = new Function<>(){

			private List<String> columns = table.getColumns();


			@Override
			public Map<String, Object> apply(org.jpmml.evaluator.Table.Row row){
				Map<String, Object> result = new LinkedHashMap<>();

				String labelCol = this.columns.get(this.columns.size() - 1);

				// XXX: Could be omitted
				result.put("label", row.get(labelCol));

				for(int i = 0; i < this.columns.size() - 1; i++){
					String featureCol = this.columns.get(i);

					result.put("features[" + i + "]", row.get(featureCol));
				}

				return result;
			}
		};

		Table result = table.stream()
			.map(rowMapper)
			.collect(new TableCollector());

		return result;
	}
}