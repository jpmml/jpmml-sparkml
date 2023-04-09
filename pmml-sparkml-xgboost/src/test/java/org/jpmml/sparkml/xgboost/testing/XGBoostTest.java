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

import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.dmg.pmml.PMML;
import org.dmg.pmml.VerificationField;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.testing.Fields;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.FloatEquivalence;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sparkml.testing.SparkMLEncoderBatch;
import org.jpmml.sparkml.testing.SparkMLEncoderBatchTest;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

public class XGBoostTest extends SparkMLEncoderBatchTest {

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
			public PMML getPMML() throws Exception {
				PMML pmml = super.getPMML();

				Visitor visitor = new AbstractVisitor(){

					@Override
					public VisitorAction visit(VerificationField verificationField){
						verificationField
							.setPrecision(1e-6d)
							.setZeroThreshold(1e-6d);

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
		evaluate("XGBoost", "Audit", excludeFields(Fields.AUDIT_PROBABILITY_FALSE));
	}

	@Test
	public void evaluateAuto() throws Exception {
		evaluate("XGBoost", "Auto");
	}

	@BeforeClass
	static
	public void createSparkSession(){
		SparkMLEncoderBatchTest.createSparkSession();
	}

	@AfterClass
	static
	public void destroySparkSession(){
		SparkMLEncoderBatchTest.destroySparkSession();
	}
}