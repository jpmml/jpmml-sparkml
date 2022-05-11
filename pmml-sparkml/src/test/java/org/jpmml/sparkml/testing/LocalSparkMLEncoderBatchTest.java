/*
 * Copyright (c) 2022 Villu Ruusmann
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
import org.jpmml.evaluator.ResultField;
import org.jpmml.sparkml.SparkSessionUtil;
import org.junit.AfterClass;
import org.junit.BeforeClass;

public class LocalSparkMLEncoderBatchTest extends SparkMLEncoderBatchTest {

	@Override
	public SparkSession getSparkSession(){
		return LocalSparkMLEncoderBatchTest.sparkSession;
	}

	@Override
	public SparkMLEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		columnFilter = excludePredictionFields(columnFilter);

		return super.createBatch(algorithm, dataset, columnFilter, equivalence);
	}

	static
	public Predicate<ResultField> excludePredictionFields(Predicate<ResultField> columnFilter){
		Predicate<ResultField> excludePredictionFields = excludeFields("prediction", FieldNameUtil.create("pmml", "prediction"));

		return columnFilter.and(excludePredictionFields);
	}

	@BeforeClass
	static
	public void createSparkSession(){
		LocalSparkMLEncoderBatchTest.sparkSession = SparkSessionUtil.createSparkSession();
	}

	@AfterClass
	static
	public void destroySparkSession(){
		LocalSparkMLEncoderBatchTest.sparkSession = SparkSessionUtil.destroySparkSession(LocalSparkMLEncoderBatchTest.sparkSession);
	}

	private static SparkSession sparkSession = null;
}