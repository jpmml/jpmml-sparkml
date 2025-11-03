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

import java.util.Map;

import scala.collection.JavaConverters;
import scala.jdk.CollectionConverters;

public class DomainUtil {

	private DomainUtil(){
	}

	static
	public <V> Map<String, V[]> toJavaMap(scala.collection.immutable.Map scalaMap){
		Map<String, V[]> javaMap = (Map)CollectionConverters.mapAsJavaMap(scalaMap);

		return javaMap;
	}

	static
	public <V> scala.collection.immutable.Map toScalaMap(Map<String, V[]> javaMap){
		scala.collection.mutable.Map scalaMap = (scala.collection.mutable.Map)JavaConverters.mapAsScalaMap(javaMap);

		return scalaMap.toMap(scala.Predef$.MODULE$.conforms());
	}
}