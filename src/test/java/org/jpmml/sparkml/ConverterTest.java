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
import java.util.Set;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ArchiveBatch;
import org.jpmml.evaluator.Batch;
import org.jpmml.evaluator.IntegrationTest;
import org.jpmml.model.SerializationUtil;

abstract
public class ConverterTest extends IntegrationTest {

	@Override
	protected ArchiveBatch createBatch(String name, String dataset){
		ArchiveBatch result = new ArchiveBatch(name, dataset){

			@Override
			public InputStream open(String path){
				Class<? extends ConverterTest> clazz = ConverterTest.this.getClass();

				return clazz.getResourceAsStream(path);
			}

			@Override
			public PMML getPMML() throws Exception {
				StructType schema = (StructType)deserialize(getDataset() + ".ser");

				PipelineModel pipelineModel = (PipelineModel)deserialize(getName() + getDataset() + ".ser");

				return ConverterUtil.toPMML(schema, pipelineModel);
			}

			private Object deserialize(String name) throws IOException, ClassNotFoundException {

				try(InputStream is = open("/ser/" + name)){
					return SerializationUtil.deserialize(is);
				}
			}
		};

		return result;
	}

	@Override
	public void evaluate(Batch batch, Set<FieldName> ignoredFields) throws Exception {
		super.evaluate(batch, ignoredFields, 1e-12, 1e-12);
	}
}