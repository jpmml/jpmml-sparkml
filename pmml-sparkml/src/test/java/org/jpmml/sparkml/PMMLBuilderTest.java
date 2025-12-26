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
package org.jpmml.sparkml;

import java.util.Arrays;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.fail;

public class PMMLBuilderTest {

	@Test
	public void construct(){
		StructType schema = new StructType();

		Model<?> model = new LogisticRegressionModel("lrm", new DenseVector(new double[0]), 0d);

		try {
			PMMLBuilder pmmlBuilder = new PMMLBuilder(schema, model);

			fail();
		} catch(SparkMLException se){
			// Ignored
		}

		PipelineModel pipelineModel = new PipelineModel("pm", Arrays.asList(model));

		try {
			PMMLBuilder pmmlBuilder = new PMMLBuilder(schema, pipelineModel);
		} catch(SparkMLException se){
			throw se;
		}
	}
}