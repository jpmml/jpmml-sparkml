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
package org.jpmml.sparkml;

import java.util.Map;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.general_regression.GeneralRegressionModel;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.ArchiveBatch;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.model.HasRegressionTableOptions;
import org.jpmml.sparkml.model.HasTreeOptions;
import org.junit.Test;

public class ClassificationTest extends SparkMLTest implements Algorithms, Datasets, Fields {

	@Override
	public ArchiveBatch createBatch(String name, String dataset, Predicate<ResultField> predicate, Equivalence<Object> equivalence){
		predicate = excludePredictionFields(predicate);

		ArchiveBatch result = new SparkMLTestBatch(name, dataset, predicate, equivalence){

			@Override
			public ClassificationTest getIntegrationTest(){
				return ClassificationTest.this;
			}

			@Override
			public Map<String, Object> getOptions(String name, String dataset){
				Map<String, Object> options = super.getOptions(name, dataset);

				if((LOGISTIC_REGRESSION).equals(name) && (AUDIT).equals(dataset)){
					options.put(HasRegressionTableOptions.OPTION_REPRESENTATION, GeneralRegressionModel.class.getSimpleName());
				} // End if

				if((DECISION_TREE).equals(name) || (GBT).equals(name) || (RANDOM_FOREST).equals(name)){
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
		evaluate(LOGISTIC_REGRESSION, AUDIT, excludeFields(AUDIT_PROBABILITY_FALSE));
	}

	@Test
	public void evaluateModelChainAudit() throws Exception {
		evaluate(MODEL_CHAIN, AUDIT);
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