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
package org.jpmml.sparkml.feature;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructType;
import org.jpmml.sparkml.SparkMLTest;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VectorDisassemblerTest extends SparkMLTest {

	@Test
	public void transform(){
		StructType schema = new StructType()
			.add("featureVec", new VectorUDT(), false);

		List<Row> rows = Arrays.asList(
			RowFactory.create(new SparseVector(3, new int[]{1}, new double[]{1.0})),
			RowFactory.create(new DenseVector(new double[]{0.0d, 0.0d, 1.0d})),
			RowFactory.create(new SparseVector(3, new int[]{0}, new double[]{1.0}))
		);

		Dataset<Row> ds = SparkMLTest.sparkSession.createDataFrame(rows, schema);

		Transformer transformer = new VectorDisassembler()
			.setInputCol("featureVec")
			.setOutputCols(new String[]{"first", "second", "third"});

		Pipeline pipeline = new Pipeline()
			.setStages(new PipelineStage[]{transformer});

		PipelineModel pipelineModel = pipeline.fit(ds);

		Dataset<Row> transformedDs = pipelineModel.transform(ds);

		checkColumn(Arrays.asList(null, 0.0d, 1.0d), transformedDs.select("first"));
		checkColumn(Arrays.asList(1.0d, 0.0d, null), transformedDs.select("second"));
		checkColumn(Arrays.asList(null, 1.0d, null), transformedDs.select("third"));
	}

	static
	private void checkColumn(List<Double> expectedValues, Dataset dataset){
		List<Row> rows = dataset.collectAsList();

		List<Double> actualValues = rows.stream()
			.map(row -> (Double)row.get(0))
			.collect(Collectors.toList());

		assertEquals(expectedValues, actualValues);
	}
}