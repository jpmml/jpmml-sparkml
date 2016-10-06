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

import org.junit.Test;

public class RegressionTest extends ConverterTest {

	@Test
	public void evaluateDecisionTreeAuto() throws Exception {
		evaluate("DecisionTree", "Auto");
	}

	@Test
	public void evaluateGBTAuto() throws Exception {
		evaluate("GBT", "Auto");
	}

	@Test
	public void evaluateGLMAuto() throws Exception {
		evaluate("GLM", "Auto");
	}

	@Test
	public void evaluateLinearRegressionAuto() throws Exception {
		evaluate("LinearRegression", "Auto");
	}

	@Test
	public void evaluateModelChainAuto() throws Exception {
		evaluate("ModelChain", "Auto");
	}

	@Test
	public void evaluateRandomForestAuto() throws Exception {
		evaluate("RandomForest", "Auto");
	}

	@Test
	public void evaluateDecisionTreeHousing() throws Exception {
		evaluate("DecisionTree", "Housing");
	}

	@Test
	public void evaluateGLMHousing() throws Exception {
		evaluate("GLM", "Housing");
	}

	@Test
	public void evaluateLinearRegressionHousing() throws Exception {
		evaluate("LinearRegression", "Housing");
	}

	@Test
	public void evaluateRandomForestHousing() throws Exception {
		evaluate("RandomForest", "Housing");
	}
}