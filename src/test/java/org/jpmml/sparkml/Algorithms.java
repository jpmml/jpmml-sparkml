/*
 * Copyright (c) 2021 Villu Ruusmann
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

interface Algorithms {

	String DECISION_TREE = "DecisionTree";
	String FP_GROWTH = "FPGrowth";
	String GBT = "GBT";
	String GLM = "GLM";
	String K_MEANS = "KMeans";
	String LINEAR_REGRESION = "LinearRegression";
	String LINEAR_SVC = "LinearSVC";
	String LOGISTIC_REGRESSION = "LogisticRegression";
	String MODEL_CHAIN = "ModelChain";
	String NAIVE_BAYES = "NaiveBayes";
	String NEURAL_NETWORK = "NeuralNetwork";
	String RANDOM_FOREST = "RandomForest";
}