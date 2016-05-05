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
	public void evaluateLinearRegressionAuto() throws Exception {
		evaluate("LinearRegression", "Auto");
	}

	@Test
	public void evaluateRandomForestAuto() throws Exception {
		evaluate("RandomForest", "Auto");
	}
}