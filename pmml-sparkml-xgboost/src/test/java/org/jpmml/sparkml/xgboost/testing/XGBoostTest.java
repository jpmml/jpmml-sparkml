/*
 * Copyright (c) 2017 Villu Ruusmann
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
package org.jpmml.sparkml.xgboost.testing;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.VerificationField;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.converter.testing.Fields;
import org.jpmml.converter.testing.OptionsUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sparkml.testing.SparkMLEncoderBatch;
import org.jpmml.sparkml.testing.SparkMLEncoderBatchTest;
import org.jpmml.xgboost.HasXGBoostOptions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class XGBoostTest extends SparkMLEncoderBatchTest implements Datasets {

	public XGBoostTest(){
		super(new FloatEquivalence(12));
	}

	@Override
	public SparkMLEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		columnFilter = columnFilter.and(excludePredictionFields());

		SparkMLEncoderBatch result = new SparkMLEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public XGBoostTest getArchiveBatchTest(){
				return XGBoostTest.this;
			}

			@Override
			public List<Map<String, Object>> getOptionsMatrix(){
				Map<String, Object> options = new LinkedHashMap<>();

				options.put(HasXGBoostOptions.OPTION_COMPACT, new Boolean[]{false, true});
				options.put(HasXGBoostOptions.OPTION_INPUT_FLOAT, new Boolean[]{false, true});
				options.put(HasXGBoostOptions.OPTION_PRUNE, false);

				return OptionsUtil.generateOptionsMatrix(options);
			}

			@Override
			public PMML getPMML() throws Exception {
				PMML pmml = super.getPMML();

				String dataset = getDataset();

				Visitor visitor = new AbstractVisitor(){

					@Override
					public VisitorAction visit(Model model){

						if(Objects.equals(dataset, AUDIT) || Objects.equals(dataset, AUDIT_NA)){
							model.setModelVerification(null);
						}

						return super.visit(model);
					}

					@Override
					public VisitorAction visit(VerificationField verificationField){
						verificationField
							.setPrecision(1e-5d)
							.setZeroThreshold(1e-5d);

						return super.visit(verificationField);
					}
				};
				visitor.applyTo(pmml);

				return pmml;
			}
		};

		return result;
	}

	@Test
	public void evaluateAudit() throws Exception {
		evaluate("XGBoost", AUDIT, excludeFields(Fields.AUDIT_PROBABILITY_FALSE), new FloatEquivalence(64 + 8));
	}

	@Test
	public void evaluateAuditNA() throws Exception {
		evaluate("XGBoost", AUDIT_NA, excludeFields(Fields.AUDIT_PROBABILITY_FALSE), new FloatEquivalence(64 + 8));
	}

	@Test
	public void evaluateAuto() throws Exception {
		evaluate("XGBoost", AUTO);
	}

	@Test
	public void evaluateAutoNA() throws Exception {
		evaluate("XGBoost", AUTO_NA);
	}

	@Test
	public void evaluateHousing() throws Exception {
		evaluate("XGBoost", HOUSING, new FloatEquivalence(12 + 12));
	}

	@Test
	public void evaluateIris() throws Exception {
		evaluate("XGBoost", IRIS, new FloatEquivalence(16));
	}

	@BeforeAll
	static
	public void createSparkSession(){
		SparkMLEncoderBatchTest.createSparkSession();
	}

	@AfterAll
	static
	public void destroySparkSession(){
		SparkMLEncoderBatchTest.destroySparkSession();
	}
}