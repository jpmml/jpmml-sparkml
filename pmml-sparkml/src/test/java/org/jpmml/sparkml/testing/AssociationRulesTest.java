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
package org.jpmml.sparkml.testing;

import java.io.File;
import java.util.List;
import java.util.function.Predicate;

import com.google.common.base.Equivalence;
import com.google.common.collect.Iterables;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.association.AssociationModel;
import org.jpmml.converter.testing.Datasets;
import org.jpmml.evaluator.ResultField;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class AssociationRulesTest extends SimpleSparkMLEncoderBatchTest implements SparkMLAlgorithms, Datasets {

	@Override
	public SparkMLEncoderBatch createBatch(String algorithm, String dataset, Predicate<ResultField> columnFilter, Equivalence<Object> equivalence){
		columnFilter = columnFilter.and(excludePredictionFields());

		SparkMLEncoderBatch result = new SparkMLEncoderBatch(algorithm, dataset, columnFilter, equivalence){

			@Override
			public AssociationRulesTest getArchiveBatchTest(){
				return AssociationRulesTest.this;
			}

			@Override
			protected Dataset<Row> loadVerificationDataset(SparkSession sparkSession, List<File> tmpResources){
				return null;
			}
		};

		return result;
	}

	@Test
	public void evaluateFPGrowthShopping() throws Exception {
		Predicate<ResultField> predicate = (resultField -> true);
		Equivalence<Object> equivalence = getEquivalence();

		try(SparkMLEncoderBatch batch = createBatch(FP_GROWTH, SHOPPING, predicate, equivalence)){
			PMML pmml = batch.getPMML();

			Model model = Iterables.getOnlyElement(pmml.getModels());

			assertTrue(model instanceof AssociationModel);
		}
	}
}