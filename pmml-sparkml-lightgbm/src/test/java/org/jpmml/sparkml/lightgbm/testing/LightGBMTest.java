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
package org.jpmml.sparkml.lightgbm.testing;

import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.testing.SparkMLEncoderBatch;
import org.jpmml.sparkml.testing.SparkMLEncoderBatchTest;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

public class LightGBMTest extends SparkMLEncoderBatchTest {

	public LightGBMTest(){
		super(new PMMLEquivalence(1e-14, 1e-14));
	}

	@Override
	public SparkMLEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		columnFilter = columnFilter.and(SparkMLEncoderBatchTest.excludePredictionFields());

		return super.createBatch(algorithm, dataset, columnFilter, equivalence);
	}

	@Test
	public void evaluateLightGBMAudit() throws Exception {
		evaluate("LightGBM", "Audit");
	}

	@Test
	public void evaluateLightGBMAuto() throws Exception {
		evaluate("LightGBM", "Auto");
	}

	@Test
	public void evaluateLightGBMIris() throws Exception {
		evaluate("LightGBM", "Iris");
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