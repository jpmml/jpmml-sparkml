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

import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.testing.Fields;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.sparkml.testing.SparkMLEncoderBatchTest;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

public class XGBoostTest extends SparkMLEncoderBatchTest {

	public XGBoostTest(){
		super(new FloatEquivalence(12));
	}

	@Override
	public SparkSession getSparkSession(){
		return XGBoostTest.sparkSession;
	}

	@Test
	public void evaluateAudit() throws Exception {
		evaluate("XGBoost", "Audit", excludeFields("prediction", FieldNameUtil.create("pmml", "prediction"), Fields.AUDIT_PROBABILITY_FALSE));
	}

	@Test
	public void evaluateAuto() throws Exception {
		evaluate("XGBoost", "Auto", excludeFields("prediction"));
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

		XGBoostTest.sparkSession = sparkSession;
	}

	@AfterClass
	static
	public void destroySparkSession(){
		XGBoostTest.sparkSession.stop();

		XGBoostTest.sparkSession = null;
	}

	private static SparkSession sparkSession = null;
}