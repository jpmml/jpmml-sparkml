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
package org.jpmml.sparkml.testing;

import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.apache.spark.sql.SparkSession;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.testing.ModelEncoderBatchTest;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.SparkSessionUtil;

public class SparkMLEncoderBatchTest extends ModelEncoderBatchTest {

	public SparkMLEncoderBatchTest(){
		this(new PMMLEquivalence(1e-14, 1e-14));
	}

	public SparkMLEncoderBatchTest(Equivalence<Object> equivalence){
		super(equivalence);
	}

	/**
	 * @see #createSparkSession()
	 * @see #destroySparkSession()
	 */
	public SparkSession getSparkSession(){
		return SparkMLEncoderBatchTest.sparkSession;
	}

	@Override
	public SparkMLEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		SparkMLEncoderBatch result = new SparkMLEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public SparkMLEncoderBatchTest getArchiveBatchTest(){
				return SparkMLEncoderBatchTest.this;
			}
		};

		return result;
	}

	static
	public void createSparkSession(){
		SparkMLEncoderBatchTest.sparkSession = SparkSessionUtil.createSparkSession();
	}

	static
	public void destroySparkSession(){
		SparkMLEncoderBatchTest.sparkSession = SparkSessionUtil.destroySparkSession(SparkMLEncoderBatchTest.sparkSession);
	}

	static
	public Predicate<ResultField> excludePredictionFields(){
		return excludePredictionFields("prediction");
	}

	static
	public Predicate<ResultField> excludePredictionFields(String predictionCol){
		return excludeFields(predictionCol, FieldNameUtil.create("pmml", predictionCol));
	}

	private static SparkSession sparkSession = null;
}