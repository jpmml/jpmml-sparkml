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

import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.general_regression.GeneralRegressionModel;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.model.HasRegressionTableOptions;
import org.jpmml.sparkml.model.HasTreeOptions;
import org.junit.Test;

public class RegressionTest extends LocalSparkMLEncoderBatchTest implements Algorithms, Datasets {

	@Override
	public SparkMLEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		columnFilter = excludePredictionFields(columnFilter);

		SparkMLEncoderBatch result = new SparkMLEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public RegressionTest getArchiveBatchTest(){
				return RegressionTest.this;
			}

			@Override
			public Map<String, Object> getOptions(){
				String algorithm = getAlgorithm();
				String dataset = getDataset();

				Map<String, Object> options = super.getOptions();

				if((LINEAR_REGRESION).equals(algorithm) && (AUTO).equals(dataset)){
					options.put(HasRegressionTableOptions.OPTION_REPRESENTATION, GeneralRegressionModel.class.getSimpleName());
				} // End if

				if((DECISION_TREE).equals(algorithm) || (GBT).equals(algorithm) || (RANDOM_FOREST).equals(algorithm)){
					options.put(HasTreeOptions.OPTION_ESTIMATE_FEATURE_IMPORTANCES, Boolean.TRUE);
				}

				return options;
			}
		};

		return result;
	}

	@Test
	public void evaluateDecisionTreeAuto() throws Exception {
		evaluate(DECISION_TREE, AUTO);
	}

	@Test
	public void evaluateGBTAuto() throws Exception {
		evaluate(GBT, AUTO);
	}

	@Test
	public void evaluateGLMAuto() throws Exception {
		evaluate(GLM, AUTO);
	}

	@Test
	public void evaluateLinearRegressionAuto() throws Exception {
		String[] transformFields = {"mpgBucket"};

		evaluate(LINEAR_REGRESION, AUTO, excludeFields(transformFields));
	}

	@Test
	public void evaluateModelChainAuto() throws Exception {
		evaluate(MODEL_CHAIN, AUTO);
	}

	@Test
	public void evaluateRandomForestAuto() throws Exception {
		evaluate(RANDOM_FOREST, AUTO);
	}

	@Test
	public void evaluateDecisionTreeHousing() throws Exception {
		evaluate(DECISION_TREE, HOUSING);
	}

	@Test
	public void evaluateGLMHousing() throws Exception {
		evaluate(GLM, HOUSING);
	}

	@Test
	public void evaluateLinearRegressionHousing() throws Exception {
		evaluate(LINEAR_REGRESION, HOUSING);
	}

	@Test
	public void evaluateRandomForestHousing() throws Exception {
		evaluate(RANDOM_FOREST, HOUSING);
	}

	@Test
	public void evaluateGLMFormulaVisit() throws Exception {
		evaluate(GLM, VISIT, new PMMLEquivalence(1e-12, 1e-12));
	}
}