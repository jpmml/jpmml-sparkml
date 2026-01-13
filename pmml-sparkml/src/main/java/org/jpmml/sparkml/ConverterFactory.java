/*
 * Copyright (c) 2018 Villu Ruusmann
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
import java.util.Arrays;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.Transformer;
import org.apache.spark.sql.SparkSession;
import org.jpmml.converter.ExceptionUtil;

public class ConverterFactory {

	private Map<RegexKey, ? extends Map<String, ?>> options = null;


	public ConverterFactory(Map<RegexKey, ? extends Map<String, ?>> options){
		setOptions(options);
	}

	public TransformerConverter<?> newConverter(Transformer transformer){
		Class<? extends Transformer> clazz = transformer.getClass();

		Class<? extends TransformerConverter<?>> converterClazz = ConverterFactory.converters.get(clazz);
		if(converterClazz == null){
			throw new SparkMLException("Transformer class " + ExceptionUtil.formatClass(clazz) + " is not supported");
		}

		TransformerConverter<?> converter;

		try {
			Constructor<? extends TransformerConverter<?>> converterConstructor = converterClazz.getDeclaredConstructor(clazz);

			converter = converterConstructor.newInstance(transformer);
		} catch(ReflectiveOperationException roe){
			throw new SparkMLException("Transformer class " + ExceptionUtil.formatClass(clazz) + " is not supported", roe);
		}

		if(converter != null){
			Map<RegexKey, ? extends Map<String, ?>> options = getOptions();

			Map<String, Object> converterOptions = new LinkedHashMap<>();

			options.entrySet().stream()
				.filter(entry -> (entry.getKey()).test(transformer.uid()))
				.map(entry -> entry.getValue())
				.forEach(converterOptions::putAll);

			converter.setOptions(converterOptions);
		}

		return converter;
	}

	public Map<RegexKey, ? extends Map<String, ?>> getOptions(){
		return this.options;
	}

	private void setOptions(Map<RegexKey, ? extends Map<String, ?>> options){
		this.options = Objects.requireNonNull(options);
	}

	static
	public void checkVersion(){
		SparkSession sparkSession;

		try {
			sparkSession = SparkSession.active();
		} catch(IllegalStateException ise){
			logger.warn("Failed to check Apache Spark ML version", ise);

			return;
		}

		SparkContext sparkContext = sparkSession.sparkContext();

		int[] version = parseVersion(sparkContext.version());

		if(!Arrays.equals(ConverterFactory.VERSION, version)){
			throw new SparkMLException("Expected Apache Spark ML version " + ExceptionUtil.formatVersion(formatVersion(ConverterFactory.VERSION)) + ", got " + ExceptionUtil.formatVersion(formatVersion(version)) + " (" + sparkContext.version() + ")");
		}
	}

	static
	public void checkNoShading(){
		Package _package = TransformerConverter.class.getPackage();

		String name = _package.getName();

		if(!(name).equals("org.jpmml.sparkml")){
			throw new SparkMLException("Expected JPMML-SparkML converter classes to have package name prefix " + ExceptionUtil.formatName("org.jpmml.sparkml") + ", got " + ExceptionUtil.formatName(name));
		}
	}

	static
	private void init(ClassLoader classLoader){
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

	@SuppressWarnings({"rawtypes", "unchecked"})
	static
	private void init(ClassLoader classLoader, Properties properties){

		if(properties.isEmpty()){
			return;
		}

		Set<String> keys = properties.stringPropertyNames();
		for(String key : keys){
			String value = properties.getProperty(key);

			logger.trace("Mapping transformer class " + key + " to transformer converter class " + value);

			Class<?> clazz;

			try {
				clazz = classLoader.loadClass(key);
			} catch(ClassNotFoundException cnfe){
				logger.warn("Failed to load transformer class", cnfe);

				continue;
			}

			if(!(Transformer.class).isAssignableFrom(clazz)){
				throw new SparkMLException("Transformer class " + ExceptionUtil.formatClass(clazz) + " is not a subclass of " + ExceptionUtil.formatClass(Transformer.class));
			}

			Class<?> converterClazz;

			try {
				converterClazz = classLoader.loadClass(value);
			} catch(ClassNotFoundException cnfe){
				logger.warn("Failed to load transformer converter class", cnfe);

				continue;
			}

			if(!(TransformerConverter.class).isAssignableFrom(converterClazz)){
				throw new SparkMLException("Transformer converter class " + ExceptionUtil.formatClass(converterClazz) + " is not a subclass of " + ExceptionUtil.formatClass(TransformerConverter.class));
			}

			ConverterFactory.converters.put((Class)clazz, (Class)converterClazz);
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

	private static final int[] VERSION = {4, 0};

	private static final Map<Class<? extends Transformer>, Class<? extends TransformerConverter<?>>> converters = new LinkedHashMap<>();

	private static final Logger logger = LogManager.getLogger(ConverterFactory.class);

	static {
		ClassLoader clazzLoader = ConverterFactory.class.getClassLoader();

		ConverterFactory.init(clazzLoader);
	}
}