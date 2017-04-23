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
import java.lang.reflect.Constructor;
import java.net.URL;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import javax.xml.bind.JAXBException;

import com.google.common.collect.Iterables;
import org.apache.commons.io.output.ByteArrayOutputStream;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.MultipleModelMethodType;
import org.dmg.pmml.Output;
import org.dmg.pmml.PMML;
import org.jpmml.converter.MiningModelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.model.MetroJAXBUtil;

public class ConverterUtil {

	private ConverterUtil(){
	}

	static
	public PMML toPMML(StructType schema, PipelineModel pipelineModel){
		FeatureMapper featureMapper = new FeatureMapper(schema);

		Map<String, org.dmg.pmml.Model> models = new LinkedHashMap<>();

		List<Transformer> transformers = getTransformers(pipelineModel);
		for(Transformer transformer : transformers){
			TransformerConverter<?> converter = ConverterUtil.createConverter(transformer);

			if(converter instanceof FeatureConverter){
				FeatureConverter<?> featureConverter = (FeatureConverter<?>)converter;

				featureMapper.append(featureConverter);
			} else

			if(converter instanceof ModelConverter){
				ModelConverter<?> modelConverter = (ModelConverter<?>)converter;

				Schema featureSchema = featureMapper.createSchema(modelConverter);

				org.dmg.pmml.Model model = modelConverter.encodeModel(featureSchema);

				featureMapper.append(modelConverter);

				HasPredictionCol hasPredictionCol = (HasPredictionCol)transformer;

				models.put(hasPredictionCol.getPredictionCol(), model);
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		org.dmg.pmml.Model rootModel;

		if(models.size() == 1){
			rootModel = Iterables.getOnlyElement(models.values());
		} else

		if(models.size() >= 2){
			List<MiningField> targetMiningFields = new ArrayList<>();

			List<Map.Entry<String, org.dmg.pmml.Model>> entries = new ArrayList<>(models.entrySet());
			for(Iterator<Map.Entry<String, org.dmg.pmml.Model>> entryIt = entries.iterator(); entryIt.hasNext(); ){
				Map.Entry<String, org.dmg.pmml.Model> entry = entryIt.next();

				String predictionCol = entry.getKey();
				org.dmg.pmml.Model model = entry.getValue();

				MiningSchema miningSchema = model.getMiningSchema();

				List<MiningField> miningFields = miningSchema.getMiningFields();
				for(Iterator<MiningField> miningFieldIt = miningFields.iterator(); miningFieldIt.hasNext(); ){
					MiningField miningField = miningFieldIt.next();

					FieldUsageType fieldUsage = miningField.getUsageType();
					switch(fieldUsage){
						case PREDICTED:
						case TARGET:
							targetMiningFields.add(miningField);
							break;
						default:
							break;
					}
				}

				if(!entryIt.hasNext()){
					break;
				}

				FieldName name = FieldName.create(predictionCol);

				featureMapper.removeDataField(name);

				Output output = model.getOutput();
				if(output == null){
					output = new Output();

					model.setOutput(output);
				}

				output.addOutputFields(ModelUtil.createPredictedField(name));
			}

			List<org.dmg.pmml.Model> memberModels = new ArrayList<>(models.values());

			MiningSchema miningSchema = new MiningSchema(targetMiningFields);

			org.dmg.pmml.Model lastMemberModel = Iterables.getLast(memberModels);

			MiningModel miningModel = new MiningModel(lastMemberModel.getFunctionName(), miningSchema)
				.setSegmentation(MiningModelUtil.createSegmentation(MultipleModelMethodType.MODEL_CHAIN, memberModels));

			rootModel = miningModel;
		} else

		{
			throw new IllegalArgumentException();
		}

		PMML pmml = featureMapper.encodePMML(rootModel);

		return pmml;
	}

	static
	public byte[] toPMMLByteArray(StructType schema, PipelineModel pipelineModel){
		PMML pmml = toPMML(schema, pipelineModel);

		ByteArrayOutputStream os = new ByteArrayOutputStream(1024 * 1024);

		try {
			MetroJAXBUtil.marshalPMML(pmml, os);
		} catch(JAXBException je){
			throw new RuntimeException(je);
		}

		return os.toByteArray();
	}

	static
	public FeatureConverter<?> createFeatureConverter(Transformer transformer){
		return (FeatureConverter<?>)createConverter(transformer);
	}

	static
	public ModelConverter<?> createModelConverter(Transformer transformer){
		return (ModelConverter<?>)createConverter(transformer);
	}

	static
	public <T extends Transformer> TransformerConverter<T> createConverter(T transformer){
		Class<? extends Transformer> clazz = transformer.getClass();

		Class<? extends TransformerConverter> converterClazz = getConverterClazz(clazz);
		if(converterClazz == null){
			throw new IllegalArgumentException("Transformer class " + clazz.getName() + " is not supported");
		}

		try {
			Constructor<?> constructor = converterClazz.getDeclaredConstructor(clazz);

			return (TransformerConverter)constructor.newInstance(transformer);
		} catch(ReflectiveOperationException roe){
			throw new IllegalArgumentException(roe);
		}
	}

	static
	public Class<? extends TransformerConverter> getConverterClazz(Class<? extends Transformer> clazz){
		return ConverterUtil.converters.get(clazz);
	}

	static
	public void putConverterClazz(Class<? extends Transformer> clazz, Class<? extends TransformerConverter<?>> converterClazz){

		if(clazz == null || !(Transformer.class).isAssignableFrom(clazz)){
			throw new IllegalArgumentException();
		} // End if

		if(converterClazz == null || !(TransformerConverter.class).isAssignableFrom(converterClazz)){
			throw new IllegalArgumentException();
		}

		ConverterUtil.converters.put(clazz, converterClazz);
	}

	static
	private List<Transformer> getTransformers(PipelineModel pipelineModel){
		List<Transformer> result = new ArrayList<>();

		Transformer[] stages = pipelineModel.stages();
		for(Transformer stage : stages){

			if(stage instanceof PipelineModel){
				PipelineModel nestedPipelineModel = (PipelineModel)stage;

				result.addAll(getTransformers(nestedPipelineModel));
			} else

			{
				result.add(stage);
			}
		}

		return result;
	}

	static
	private void init(){
		Thread thread = Thread.currentThread();

		ClassLoader classLoader = thread.getContextClassLoader();
		if(classLoader == null){
			classLoader = ClassLoader.getSystemClassLoader();
		}

		Enumeration<URL> urls;

		try {
			urls = classLoader.getResources("META-INF/sparkml2pmml.properties");
		} catch(IOException ioe){
			logger.warn("Failed to find resources", ioe);

			return;
		}

		while(urls.hasMoreElements()){
			URL url = urls.nextElement();

			logger.trace("Loading resource " + url);

			try(InputStream is = url.openStream()){
				Properties properties = new Properties();
				properties.load(is);

				init(classLoader, properties);
			} catch(IOException ioe){
				logger.warn("Failed to load resource", ioe);
			}
		}
	}

	static
	private void init(ClassLoader classLoader, Properties properties){

		if(properties.isEmpty()){
			return;
		}

		Set<String> keys = properties.stringPropertyNames();
		for(String key : keys){
			String value = properties.getProperty(key);

			logger.trace("Mapping transformer class " + key + " to transformer converter class " + value);

			Class<? extends Transformer> clazz;

			try {
				clazz = (Class)classLoader.loadClass(key);
			} catch(ClassNotFoundException cnfe){
				logger.warn("Failed to load transformer class", cnfe);

				continue;
			}

			Class<? extends TransformerConverter<?>> converterClazz;

			try {
				converterClazz = (Class)classLoader.loadClass(value);
			} catch(ClassNotFoundException cnfe){
				logger.warn("Failed to load transformer converter class", cnfe);

				continue;
			}

			putConverterClazz(clazz, converterClazz);
		}
	}

	private static final Map<Class<? extends Transformer>, Class<? extends TransformerConverter>> converters = new LinkedHashMap<>();

	private static final Logger logger = LogManager.getLogger(ConverterUtil.class);

	static {
		ConverterUtil.init();
	}
}