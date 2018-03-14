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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.xml.bind.JAXBException;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.PMML;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.MetroJAXBUtil;

public class ConverterUtil {

	private ConverterUtil(){
	}

	static
	public PMML toPMML(StructType schema, PipelineModel pipelineModel){
		checkVersion();

		SparkMLEncoder encoder = new SparkMLEncoder(schema);

		List<org.dmg.pmml.Model> models = new ArrayList<>();

		Iterable<Transformer> transformers = getTransformers(pipelineModel);
		for(Transformer transformer : transformers){
			TransformerConverter<?> converter = ConverterUtil.createConverter(transformer);

			if(converter instanceof FeatureConverter){
				FeatureConverter<?> featureConverter = (FeatureConverter<?>)converter;

				featureConverter.registerFeatures(encoder);
			} else

			if(converter instanceof ModelConverter){
				ModelConverter<?> modelConverter = (ModelConverter<?>)converter;

				org.dmg.pmml.Model model = modelConverter.registerModel(encoder);

				models.add(model);
			} else

			{
				throw new IllegalArgumentException("Expected a " + FeatureConverter.class.getName() + " or " + ModelConverter.class.getName() + " instance, got " + converter);
			}
		}

		org.dmg.pmml.Model rootModel;

		if(models.size() == 1){
			rootModel = Iterables.getOnlyElement(models);
		} else

		if(models.size() > 1){
			List<MiningField> targetMiningFields = new ArrayList<>();

			for(org.dmg.pmml.Model model : models){
				MiningSchema miningSchema = model.getMiningSchema();

				List<MiningField> miningFields = miningSchema.getMiningFields();
				for(MiningField miningField : miningFields){
					MiningField.UsageType usageType = miningField.getUsageType();

					switch(usageType){
						case PREDICTED:
						case TARGET:
							targetMiningFields.add(miningField);
							break;
						default:
							break;
					}
				}
			}

			MiningSchema miningSchema = new MiningSchema(targetMiningFields);

			MiningModel miningModel = MiningModelUtil.createModelChain(models, new Schema(null, Collections.<Feature>emptyList()))
				.setMiningSchema(miningSchema);

			rootModel = miningModel;
		} else

		{
			throw new IllegalArgumentException("Expected a pipeline with one or more models, got a pipeline with zero models");
		}

		PMML pmml = encoder.encodePMML(rootModel);

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
			throw new IllegalArgumentException("Expected " + Transformer.class.getName() + " subclass, got " + (clazz != null ? clazz.getName() : null));
		} // End if

		if(converterClazz == null || !(TransformerConverter.class).isAssignableFrom(converterClazz)){
			throw new IllegalArgumentException("Expected " + TransformerConverter.class.getName() + " subclass, got " + (converterClazz != null ? converterClazz.getName() : null));
		}

		ConverterUtil.converters.put(clazz, converterClazz);
	}

	static
	private Iterable<Transformer> getTransformers(PipelineModel pipelineModel){
		List<Transformer> transformers = new ArrayList<>();
		transformers.add(pipelineModel);

		Function<Transformer, List<Transformer>> function = new Function<Transformer, List<Transformer>>(){

			@Override
			public List<Transformer> apply(Transformer transformer){

				if(transformer instanceof PipelineModel){
					PipelineModel pipelineModel = (PipelineModel)transformer;

					return Arrays.asList(pipelineModel.stages());
				} else

				if(transformer instanceof CrossValidatorModel){
					CrossValidatorModel crossValidatorModel = (CrossValidatorModel)transformer;

					return Collections.<Transformer>singletonList(crossValidatorModel.bestModel());
				} else

				if(transformer instanceof TrainValidationSplitModel){
					TrainValidationSplitModel trainValidationSplitModel = (TrainValidationSplitModel)transformer;

					return Collections.<Transformer>singletonList(trainValidationSplitModel.bestModel());
				}

				return null;
			}
		};

		while(true){
			ListIterator<Transformer> transformerIt = transformers.listIterator();

			boolean modified = false;

			while(transformerIt.hasNext()){
				Transformer transformer = transformerIt.next();

				List<Transformer> childTransformers = function.apply(transformer);
				if(childTransformers != null){
					transformerIt.remove();

					for(Transformer childTransformer : childTransformers){
						transformerIt.add(childTransformer);
					}

					modified = true;
				}
			}

			if(!modified){
				break;
			}
		}

		return transformers;
	}

	static
	public void checkVersion(){
		SparkContext sparkContext = SparkContext.getOrCreate();

		int[] version = parseVersion(sparkContext.version());

		if(!Arrays.equals(ConverterUtil.VERSION, version)){
			throw new IllegalArgumentException("Expected Apache Spark ML version " + formatVersion(ConverterUtil.VERSION) + ", got version " + formatVersion(version) + " (" + sparkContext.version() + ")");
		}
	}

	static
	private int[] parseVersion(String string){
		Pattern pattern = Pattern.compile("^(\\d+)\\.(\\d+)(\\..*)?$");

		Matcher matcher = pattern.matcher(string);
		if(!matcher.matches()){
			return new int[]{-1, -1};
		}

		return new int[]{Integer.parseInt(matcher.group(1)), Integer.parseInt(matcher.group(2))};
	}

	static
	private String formatVersion(int[] version){
		return String.valueOf(version[0]) + "." + String.valueOf(version[1]);
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

	private static final int[] VERSION = {2, 3};

	private static final Map<Class<? extends Transformer>, Class<? extends TransformerConverter>> converters = new LinkedHashMap<>();

	private static final Logger logger = LogManager.getLogger(ConverterUtil.class);

	static {
		ConverterUtil.init();
	}
}