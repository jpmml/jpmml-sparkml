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

import java.lang.reflect.Constructor;

import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.Transformer;

public class ConverterUtil {

	private ConverterUtil(){
	}

	static
	public <T extends Transformer> TransformerConverter<T> createConverter(T transformer) throws Exception {
		Class<? extends Transformer> clazz = transformer.getClass();

		Class<?> transformerClazz;

		if(transformer instanceof PredictionModel){
			transformerClazz = Class.forName("org.jpmml.sparkml.model." + clazz.getSimpleName() + "Converter");
		} else

		{
			transformerClazz = Class.forName("org.jpmml.sparkml.feature." + clazz.getSimpleName() + "Converter");
		}

		Constructor<?> constructor = transformerClazz.getDeclaredConstructor(clazz);

		return (TransformerConverter)constructor.newInstance(transformer);
	}
}