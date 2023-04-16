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

import java.util.Map;

import org.apache.spark.ml.Transformer;

abstract
public class TransformerConverter<T extends Transformer> {

	private T object = null;

	private Map<String, ?> options = null;


	public TransformerConverter(T object){
		setObject(object);
	}

	public Object getOption(String key, Object defaultValue){
		Map<String, ?> options = getOptions();

		if(options != null && options.containsKey(key)){
			return options.get(key);
		}

		return defaultValue;
	}

	public T getObject(){
		return this.object;
	}

	private void setObject(T object){
		this.object = object;
	}

	public Map<String, ?> getOptions(){
		return this.options;
	}

	public void setOptions(Map<String, ?> options){
		this.options = options;
	}
}