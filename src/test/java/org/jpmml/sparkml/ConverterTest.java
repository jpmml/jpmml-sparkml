package org.jpmml.sparkml;

import java.io.InputStream;
import java.io.ObjectInputStream;

import org.apache.spark.ml.PipelineModel;
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
				PipelineModel pipelineModel;

				try(InputStream is = open("/ser/" + getName() + getDataset() + ".ser")){

					try(ObjectInputStream ois = new ObjectInputStream(is)){
						pipelineModel = (PipelineModel)ois.readObject();
					}
				}

				return PipelineModelUtil.toPMML(pipelineModel);
			}
		};

		return result;
	}
}