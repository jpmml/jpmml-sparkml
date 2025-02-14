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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.OptionsUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.lightgbm.HasLightGBMOptions;
import org.jpmml.sparkml.testing.SparkMLEncoderBatch;
import org.jpmml.sparkml.testing.SparkMLEncoderBatchTest;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class LightGBMTest extends SparkMLEncoderBatchTest implements Datasets {

	public LightGBMTest(){
		super(new PMMLEquivalence(1e-14, 1e-14));
	}

	@Override
	public SparkMLEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		columnFilter = columnFilter.and(SparkMLEncoderBatchTest.excludePredictionFields());

		SparkMLEncoderBatch result = new SparkMLEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public LightGBMTest getArchiveBatchTest(){
				return LightGBMTest.this;
			}

			@Override
			public List<Map<String, Object>> getOptionsMatrix(){
				Map<String, Object> options = new LinkedHashMap<>();

				options.put(HasLightGBMOptions.OPTION_COMPACT, new Boolean[]{false, true});

				return OptionsUtil.generateOptionsMatrix(options);
			}
		};

		return result;
	}

	@Test
	public void evaluateLightGBMAudit() throws Exception {
		evaluate("LightGBM", AUDIT);
	}

	@Test
	public void evaluateLightGBMAuditNA() throws Exception {
		evaluate("LightGBM", AUDIT_NA);
	}

	@Test
	public void evaluateLightGBMAuto() throws Exception {
		evaluate("LightGBM", AUTO);
	}

	@Test
	public void evaluateLightGBMAutoNA() throws Exception {
		evaluate("LightGBM", AUTO_NA);
	}

	@Test
	public void evaluateLightGBMIris() throws Exception {
		evaluate("LightGBM", IRIS);
	}

	@BeforeAll
	static
	public void createSparkSession(){
		SparkMLEncoderBatchTest.createSparkSession();
	}

	@AfterAll
	static
	public void destroySparkSession(){
		SparkMLEncoderBatchTest.destroySparkSession();
	}
}