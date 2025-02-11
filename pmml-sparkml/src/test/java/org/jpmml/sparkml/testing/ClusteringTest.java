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

import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.testing.Datasets;
import org.junit.jupiter.api.Test;

public class ClusteringTest extends SimpleSparkMLEncoderBatchTest implements SparkMLAlgorithms, Datasets {

	@Test
	public void evaluateKMeansIris() throws Exception {
		String[] outputFields = {FieldNameUtil.create("pmml", "cluster")};

		evaluate(K_MEANS, IRIS, excludeFields(outputFields));
	}
}