package org.jpmml.sparkml;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ArchiveBatch;
import org.jpmml.evaluator.IntegrationTest;

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

				return PipelineModelUtil.toPMML(schema, pipelineModel);
			}

			private Object deserialize(String name) throws IOException, ClassNotFoundException {

				try(InputStream is = open("/ser/" + name)){

					try(ObjectInputStream ois = new ObjectInputStream(is)){
						return ois.readObject();
					}
				}
			}
		};

		return result;
	}
}