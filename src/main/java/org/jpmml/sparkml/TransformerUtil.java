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

import java.lang.reflect.Field;

import org.apache.spark.ml.Transformer;

public class TransformerUtil {

	private TransformerUtil(){
	}

	static
	public Object getParam(Transformer transformer, String name){
		Field field = getParamField(transformer, name);

		if(!field.isAccessible()){
			field.setAccessible(true);
		}

		try {
			return field.get(transformer);
		} catch(IllegalAccessException iae){
			throw new IllegalArgumentException(name, iae);
		}
	}

	static
	private Field getParamField(Transformer transformer, String name){
		Class<?> clazz = transformer.getClass();

		String[] prefixes = {"", (clazz.getName()).replace('.', '$') + "$$"};
		for(String prefix : prefixes){

			try {
				return clazz.getDeclaredField(prefix + name);
			} catch(NoSuchFieldException nsfe){
				// Ignored
			}
		}

		throw new IllegalArgumentException(name);
	}
}