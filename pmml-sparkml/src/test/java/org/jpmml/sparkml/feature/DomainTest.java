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

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.jpmml.sparkml.SparkMLTest;
import scala.collection.JavaConverters;
import scala.jdk.CollectionConverters;

import static org.junit.jupiter.api.Assertions.assertEquals;

abstract
public class DomainTest extends SparkMLTest {

	static
	protected void checkDataset(Map<String, List<Object>> expectedColumns, Dataset<Row> actualDs){
		Set<String> keys = expectedColumns.keySet();

		for(String key : keys){
			List<Object> expectedColumn = expectedColumns.get(key);

			List<Row> actualColumnRows = actualDs
				.select(key)
				.collectAsList();

			List<Object> actualColumn = actualColumnRows.stream()
				.map(row -> row.get(0))
				.collect(Collectors.toList());

			assertEquals(expectedColumn, actualColumn);
		}
	}

	static
	protected <V> Map<String, V[]> toJavaMap(scala.collection.immutable.Map scalaMap){
		Map<String, V[]> javaMap = (Map)CollectionConverters.mapAsJavaMap(scalaMap);

		return javaMap;
	}

	static
	protected <V> scala.collection.immutable.Map toScalaMap(Map<String, V[]> javaMap){
		scala.collection.mutable.Map scalaMap = (scala.collection.mutable.Map)JavaConverters.mapAsScalaMap(javaMap);

		return scalaMap.toMap(scala.Predef$.MODULE$.conforms());
	}
}