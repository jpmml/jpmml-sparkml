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

import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import org.apache.spark.sql.SparkSession;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.evaluator.ResultField;
import org.jpmml.evaluator.testing.ArchiveBatch;
import org.jpmml.evaluator.testing.IntegrationTest;
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.junit.AfterClass;
import org.junit.BeforeClass;

abstract
public class SparkMLTest extends IntegrationTest {

	public SparkMLTest(){
		super(new PMMLEquivalence(1e-14, 1e-14));
	}

	@Override
	protected ArchiveBatch createBatch(String name, String dataset, Predicate<ResultField> predicate, Equivalence<Object> equivalence){
		predicate = excludePredictionFields(predicate);

		ArchiveBatch result = new SparkMLTestBatch(name, dataset, predicate, equivalence){

			@Override
			public SparkMLTest getIntegrationTest(){
				return SparkMLTest.this;
			}
		};

		return result;
	}

	@BeforeClass
	static
	public void createSparkSession(){
		SparkMLTest.sparkSession = SparkSessionUtil.createSparkSession();
	}

	@AfterClass
	static
	public void destroySparkSession(){
		SparkMLTest.sparkSession = SparkSessionUtil.destroySparkSession(SparkMLTest.sparkSession);
	}

	static
	public Predicate<ResultField> excludePredictionFields(Predicate<ResultField> predicate){
		Predicate<ResultField> excludePredictionFields = excludeFields("prediction", FieldNameUtil.create("pmml", "prediction"));

		return predicate.and(excludePredictionFields);
	}

	public static SparkSession sparkSession = null;
}