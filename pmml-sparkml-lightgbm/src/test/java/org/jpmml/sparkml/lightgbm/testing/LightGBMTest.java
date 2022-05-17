/*
 * Copyright (c) 2022 Villu Ruusmann
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
package org.jpmml.sparkml.lightgbm.testing;

import org.jpmml.converter.FieldNameUtil;
import org.jpmml.evaluator.testing.IntegrationTest;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.junit.Test;

public class LightGBMTest extends IntegrationTest {

	public LightGBMTest(){
		super(new PMMLEquivalence(1e-14, 1e-14));
	}

	@Test
	public void evaluateLightGBMAudit() throws Exception {
		evaluate("LightGBM", "Audit", excludeFields("prediction", FieldNameUtil.create("pmml", "prediction")));
	}

	@Test
	public void evaluateLightGBMAuto() throws Exception {
		evaluate("LightGBM", "Auto", excludeFields("prediction"));
	}
}