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

import org.dmg.pmml.general_regression.GeneralRegressionModel;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.model.HasRegressionTableOptions;
import org.junit.Test;

public class ClassificationTest extends ConverterTest {

	@Override
	public Map<String, Object> getOptions(String name, String dataset){
		Map<String, Object> options = super.getOptions(name, dataset);

		if(("LogisticRegression").equals(name) && ("Audit").equals(dataset)){
			options.put(HasRegressionTableOptions.OPTION_REPRESENTATION, GeneralRegressionModel.class.getSimpleName());
		}

		return options;
	}

	@Test
	public void evaluateDecisionTreeAudit() throws Exception {
		evaluate("DecisionTree", "Audit");
	}

	@Test
	public void evaluateGBTAudit() throws Exception {
		evaluate("GBT", "Audit");
	}

	@Test
	public void evaluateGLMAudit() throws Exception {
		evaluate("GLM", "Audit");
	}

	@Test
	public void evaluateLogisticRegressionAudit() throws Exception {
		evaluate("LogisticRegression", "Audit", new PMMLEquivalence(5e-10, 5e-10));
	}

	@Test
	public void evaluateModelChainAudit() throws Exception {
		evaluate("ModelChain", "Audit");
	}

	@Test
	public void evaluateNaiveBayesAudit() throws Exception {
		evaluate("NaiveBayes", "Audit", new PMMLEquivalence(5e-10, 5e-10));
	}

	@Test
	public void evaluateNeuralNetworkAudit() throws Exception {
		evaluate("NeuralNetwork", "Audit");
	}

	@Test
	public void evaluateRandomForestAudit() throws Exception {
		evaluate("RandomForest", "Audit");
	}

	@Test
	public void evaluateDecisionTreeIris() throws Exception {
		evaluate("DecisionTree", "Iris");
	}

	@Test
	public void evaluateLogisticRegressionIris() throws Exception {
		evaluate("LogisticRegression", "Iris");
	}

	@Test
	public void evaluateModelChainIris() throws Exception {
		evaluate("ModelChain", "Iris");
	}

	@Test
	public void evaluateNaiveBayesIris() throws Exception {
		evaluate("NaiveBayes", "Iris");
	}

	@Test
	public void evaluateNeuralNetworkIris() throws Exception {
		evaluate("NeuralNetwork", "Iris", new PMMLEquivalence(1e-13, 1e-13));
	}

	@Test
	public void evaluateRandomForestIris() throws Exception {
		evaluate("RandomForest", "Iris");
	}

	@Test
	public void evaluateDecisionTreeSentiment() throws Exception {
		evaluate("DecisionTree", "Sentiment");
	}

	@Test
	public void evaluateGLMSentiment() throws Exception {
		evaluate("GLM", "Sentiment");
	}

	@Test
	public void evaluateLinearSVCSentiment() throws Exception {
		evaluate("LinearSVC", "Sentiment");
	}

	@Test
	public void evaluateRandomForestSentiment() throws Exception {
		evaluate("RandomForest", "Sentiment");
	}
}