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

import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.association.AssociationModel;
import org.jpmml.evaluator.ResultField;
import org.junit.Test;
import org.spark_project.guava.collect.Iterables;

import static org.junit.Assert.assertTrue;

public class AssociationRulesTest extends SparkMLTest implements Algorithms, Datasets {

	@Test
	public void evaluateFPGrowthShopping() throws Exception {
		Predicate<ResultField> predicate = (resultField -> true);
		Equivalence<Object> equivalence = getEquivalence();

		try(SparkMLTestBatch batch = (SparkMLTestBatch)createBatch(FP_GROWTH, SHOPPING, predicate, equivalence)){
			PMML pmml = batch.getPMML();

			Model model = Iterables.getOnlyElement(pmml.getModels());

			assertTrue(model instanceof AssociationModel);
		}
	}
}