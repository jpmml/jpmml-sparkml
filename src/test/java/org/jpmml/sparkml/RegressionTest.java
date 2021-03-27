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
import org.dmg.pmml.FieldName;
import org.dmg.pmml.general_regression.GeneralRegressionModel;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.ArchiveBatch;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.sparkml.model.HasRegressionTableOptions;
import org.jpmml.sparkml.model.HasTreeOptions;
import org.junit.Test;

public class RegressionTest extends SparkMLTest {

	@Override
	public ArchiveBatch createBatch(String name, String dataset, Predicate<ResultField> predicate, Equivalence<Object> equivalence){
		predicate = excludePredictionFields(predicate);

		ArchiveBatch result = new SparkMLTestBatch(name, dataset, predicate, equivalence){

			@Override
			public RegressionTest getIntegrationTest(){
				return RegressionTest.this;
			}

			@Override
			public Map<String, Object> getOptions(String name, String dataset){
				Map<String, Object> options = super.getOptions(name, dataset);

				if(("LinearRegression").equals(name) && ("Auto").equals(dataset)){
					options.put(HasRegressionTableOptions.OPTION_REPRESENTATION, GeneralRegressionModel.class.getSimpleName());
				} // End if

				if(("DecisionTree").equals(name) || ("GBT").equals(name) || ("RandomForest").equals(name)){
					options.put(HasTreeOptions.OPTION_ESTIMATE_FEATURE_IMPORTANCES, Boolean.TRUE);
				}

				return options;
			}
		};

		return result;
	}

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
		FieldName[] transformFields = {FieldName.create("mpgBucket")};

		evaluate("LinearRegression", "Auto", excludeFields(transformFields));
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

	@Test
	public void evaluateGLMFormulaVisit() throws Exception {
		evaluate("GLM", "Visit", new PMMLEquivalence(1e-12, 1e-12));
	}
}