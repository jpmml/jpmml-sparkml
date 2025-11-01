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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.jpmml.sparkml.SparkMLTest;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CategoricalDomainTest extends DomainTest {

	@Test
	public void fitTransform(){
		StructType schema = new StructType()
			.add("fruit", DataTypes.StringType, true)
			.add("color", DataTypes.StringType, true);

		List<Row> rows = Arrays.asList(
			RowFactory.create("apple", "red"),
			RowFactory.create("apple", null),
			RowFactory.create("orange", "orange"),
			RowFactory.create("banana", "yellow"),
			RowFactory.create("banana", "green"),
			RowFactory.create("apple", "green"),
			RowFactory.create(null, "pink")
		);

		Dataset<Row> ds = SparkMLTest.sparkSession.createDataFrame(rows, schema);

		Map<String, Object[]> dataValues = Collections.emptyMap();

		Map<String, List<Object>> expectedColumns;

		CategoricalDomain domain = (CategoricalDomain)new CategoricalDomain()
			.setWithData(false)
			.setInputCols(new String[]{"fruit", "color"})
			.setOutputCols(new String[]{"fruit_pmml", "color_pmml"});

		assertEquals("asIs", domain.getMissingValueTreatment());
		assertEquals("returnInvalid", domain.getInvalidValueTreatment());

		CategoricalDomainModel domainModel = domain.fit(ds);

		Dataset<Row> transformedDs = domainModel.transform(ds);

		checkDataValues(dataValues, DomainUtil.toJavaMap(domainModel.getDataValues()));

		dataValues = Map.of(
			"fruit", new Object[]{"apple", "banana", "orange"},
			"color", new Object[]{"green", "orange", "pink", "red", "yellow"}
		);

		domain = (CategoricalDomain)domain
			.setWithData(true);

		domainModel = domain.fit(ds);

		transformedDs = domainModel.transform(ds);

		checkDataValues(dataValues, DomainUtil.toJavaMap(domainModel.getDataValues()));

		dataValues = Map.of(
			"fruit", new Object[]{"apple", "orange"},
			"color", new Object[]{"green", "red", "yellow"}
		);

		expectedColumns = Map.of(
			"fruit_pmml", Arrays.asList("apple", "apple", "orange", null, null, "apple", null),
			"color_pmml", Arrays.asList("red", null, null, "yellow", "green", "green", null)
		);

		domain = (CategoricalDomain)domain
			.setDataValues(DomainUtil.toScalaMap(dataValues))
			.setInvalidValueTreatment("asMissing");

		domainModel = domain.fit(ds);

		transformedDs = domainModel.transform(ds);

		checkDataValues(dataValues, DomainUtil.toJavaMap(domainModel.getDataValues()));

		checkDataset(expectedColumns, transformedDs);

		domain = (CategoricalDomain)domain
			.setInvalidValueTreatment("asValue")
			.setInvalidValueReplacement("(invalid)");

		expectedColumns = Map.of(
			"fruit_pmml", Arrays.asList("apple", "apple", "orange", "(invalid)", "(invalid)", "apple", null),
			"color_pmml", Arrays.asList("red", null, "(invalid)", "yellow", "green", "green", "(invalid)")
		);

		domainModel = domain.fit(ds);

		transformedDs = domainModel.transform(ds);

		checkDataset(expectedColumns, transformedDs);

		domain = (CategoricalDomain)domain
			.setMissingValueTreatment("asValue")
			.setMissingValueReplacement("(missing)")
			.setInvalidValueTreatment("asMissing")
			.setInvalidValueReplacement(null);

		expectedColumns = Map.of(
			"fruit_pmml", Arrays.asList("apple", "apple", "orange", "(missing)", "(missing)", "apple", "(missing)"),
			"color_pmml", Arrays.asList("red", "(missing)", "(missing)", "yellow", "green", "green", "(missing)")
		);

		domainModel = domain.fit(ds);

		transformedDs = domainModel.transform(ds);

		checkDataset(expectedColumns, transformedDs);
	}

	static
	private void checkDataValues(Map<String, Object[]> expected, Map<String, Object[]> actual){
		Set<String> keys = expected.keySet();

		assertEquals(keys, actual.keySet());

		for(String key : keys){
			List<Object> expectedValues = new ArrayList<>(Arrays.asList(expected.get(key)));
			List<Object> actualValues = new ArrayList<>(Arrays.asList(actual.get(key)));

			Collections.sort((List)expectedValues);
			Collections.sort((List)actualValues);

			assertEquals(expectedValues, actualValues);
		}
	}
}