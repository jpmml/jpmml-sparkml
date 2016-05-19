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