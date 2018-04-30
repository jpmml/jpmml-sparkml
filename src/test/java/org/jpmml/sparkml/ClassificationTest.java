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

import org.jpmml.evaluator.PMMLEquivalence;
import org.junit.Test;

public class ClassificationTest extends ConverterTest {

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
		evaluate("LogisticRegression", "Audit", new PMMLEquivalence(2e-11, 1e-14));
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
	public void evaluateGLMSentiment() throws Exception {
		evaluate("GLM", "Sentiment");
	}

	@Test
	public void evaluateRandomForestSentiment() throws Exception {
		evaluate("RandomForest", "Sentiment");
	}
}