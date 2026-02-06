/*
 * Copyright (c) 2025 Villu Ruusmann
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
package org.jpmml.sparkml.feature;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import scala.jdk.javaapi.CollectionConverters;

public class DomainUtil {

	private DomainUtil(){
	}

	static
	public <E> E[] toArray(List<?> values, Class<E> clazz){

		if(values == null){
			return null;
		}

		@SuppressWarnings("unchecked")
		E[] result = (E[])Array.newInstance(clazz, values.size());

		return values.toArray(result);
	}

	static
	public <E> Map<String, E[]> toArrayMap(Map<String, List<?>> map, Class<E> clazz){
		Collection<Map.Entry<String, List<?>>> entries = map.entrySet();

		return entries.stream()
			.collect(Collectors.toMap(entry -> entry.getKey(), entry -> toArray(entry.getValue(), clazz), (left, right) -> left, LinkedHashMap::new));
	}

	static
	public Map<String, Object[]> toObjectArrayMap(Map<String, List<?>> map){
		return toArrayMap(map, Object.class);
	}

	static
	public Map<String, Number[]> toNumberArrayMap(Map<String, List<?>> map){
		return toArrayMap(map, Number.class);
	}

	static
	public <E> Map<String, List<E>> toListMap(Map<String, E[]> map){
		Collection<Map.Entry<String, E[]>> entries = map.entrySet();

		return entries.stream()
			.collect(Collectors.toMap(entry -> entry.getKey(), entry -> Arrays.asList(entry.getValue()), (left, right) -> left, LinkedHashMap::new));
	}

	static
	public <V> Map<String, V[]> toJavaMap(scala.collection.immutable.Map scalaMap){
		Map<String, V[]> javaMap = CollectionConverters.asJava(scalaMap);

		return javaMap;
	}

	static
	public <V> scala.collection.immutable.Map toScalaMap(Map<String, V[]> javaMap){
		scala.collection.mutable.Map scalaMap = CollectionConverters.asScala(javaMap);

		return scala.collection.immutable.Map$.MODULE$.from(scalaMap);
	}
}