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
import org.jpmml.converter.testing.Fields;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.model.HasRegressionTableOptions;
import org.jpmml.sparkml.model.HasTreeOptions;
import org.junit.Test;

public class ClassificationTest extends SimpleSparkMLEncoderBatchTest implements SparkMLAlgorithms, Datasets, Fields {

	@Override
	public SparkMLEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		columnFilter = columnFilter.and(excludePredictionFields());

		SparkMLEncoderBatch result = new SparkMLEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public ClassificationTest getArchiveBatchTest(){
				return ClassificationTest.this;
			}

			@Override
			public Map<String, Object> getOptions(){
				String algorithm = getAlgorithm();
				String dataset = getDataset();

				Map<String, Object> options = super.getOptions();

				if((LOGISTIC_REGRESSION).equals(algorithm) && (AUDIT).equals(dataset)){
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
	public void evaluateDecisionTreeAudit() throws Exception {
		evaluate(DECISION_TREE, AUDIT);
	}

	@Test
	public void evaluateGBTAudit() throws Exception {
		evaluate(GBT, AUDIT);
	}

	@Test
	public void evaluateGLMAudit() throws Exception {
		evaluate(GLM, AUDIT);
	}

	@Test
	public void evaluateLogisticRegressionAudit() throws Exception {
		evaluate(LOGISTIC_REGRESSION, AUDIT, excludeFields(AUDIT_PROBABILITY_FALSE), new PMMLEquivalence(1e-8, 1e-8));
	}

	@Test
	public void evaluateModelChainAudit() throws Exception {
		evaluate(MODEL_CHAIN, AUDIT, new PMMLEquivalence(5e-13, 5e-13));
	}

	@Test
	public void evaluateNaiveBayesAudit() throws Exception {
		evaluate(NAIVE_BAYES, AUDIT, excludeFields(AUDIT_PROBABILITY_FALSE), new PMMLEquivalence(1e-10, 1e-10));
	}

	@Test
	public void evaluateNeuralNetworkAudit() throws Exception {
		evaluate(NEURAL_NETWORK, AUDIT);
	}

	@Test
	public void evaluateRandomForestAudit() throws Exception {
		evaluate(RANDOM_FOREST, AUDIT);
	}

	@Test
	public void evaluateDecisionTreeIris() throws Exception {
		evaluate(DECISION_TREE, IRIS);
	}

	@Test
	public void evaluateLogisticRegressionIris() throws Exception {
		evaluate(LOGISTIC_REGRESSION, IRIS);
	}

	@Test
	public void evaluateModelChainIris() throws Exception {
		evaluate(MODEL_CHAIN, IRIS);
	}

	@Test
	public void evaluateNaiveBayesIris() throws Exception {
		evaluate(NAIVE_BAYES, IRIS);
	}

	@Test
	public void evaluateNeuralNetworkIris() throws Exception {
		evaluate(NEURAL_NETWORK, IRIS, new PMMLEquivalence(5e-13, 5e-13));
	}

	@Test
	public void evaluateRandomForestIris() throws Exception {
		evaluate(RANDOM_FOREST, IRIS);
	}

	@Test
	public void evaluateDecisionTreeSentiment() throws Exception {
		evaluate(DECISION_TREE, SENTIMENT);
	}

	@Test
	public void evaluateGLMSentiment() throws Exception {
		evaluate(GLM, SENTIMENT);
	}

	@Test
	public void evaluateLinearSVCSentiment() throws Exception {
		evaluate(LINEAR_SVC, SENTIMENT);
	}

	@Test
	public void evaluateRandomForestSentiment() throws Exception {
		evaluate(RANDOM_FOREST, SENTIMENT);
	}
}