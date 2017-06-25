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

import java.io.IOException;
import java.io.InputStream;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ArchiveBatch;
import org.jpmml.evaluator.IntegrationTest;
import org.jpmml.evaluator.IntegrationTestBatch;
import org.jpmml.evaluator.PMMLEquivalence;
import org.jpmml.model.SerializationUtil;

abstract
public class ConverterTest extends IntegrationTest {

	public ConverterTest(){
		super(new PMMLEquivalence(1e-12, 1e-12));
	}

	@Override
	protected ArchiveBatch createBatch(String name, String dataset, Predicate<FieldName> predicate){
		Predicate<FieldName> excludePredictionFields = excludeFields(FieldName.create("prediction"), FieldName.create("pmml(prediction)"));

		if(predicate == null){
			predicate = excludePredictionFields;
		} else

		{
			predicate = Predicates.and(predicate, excludePredictionFields);
		}

		ArchiveBatch result = new IntegrationTestBatch(name, dataset, predicate){

			@Override
			public IntegrationTest getIntegrationTest(){
				return ConverterTest.this;
			}

			@Override
			public PMML getPMML() throws Exception {
				StructType schema = (StructType)deserialize(getDataset() + ".ser");

				PipelineModel pipelineModel = (PipelineModel)deserialize(getName() + getDataset() + ".ser");

				PMML pmml = ConverterUtil.toPMML(schema, pipelineModel);

				ensureValidity(pmml);

				return pmml;
			}

			private Object deserialize(String name) throws IOException, ClassNotFoundException {

				try(InputStream is = open("/ser/" + name)){
					return SerializationUtil.deserialize(is);
				}
			}
		};

		return result;
	}
}