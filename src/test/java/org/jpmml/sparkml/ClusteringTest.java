package org.jpmml.sparkml;

import org.junit.Test;

public class ClusteringTest extends ConverterTest {

	@Test
	public void evaluateKMeansIris() throws Exception {
		evaluate("KMeans", "Iris");
	}
}