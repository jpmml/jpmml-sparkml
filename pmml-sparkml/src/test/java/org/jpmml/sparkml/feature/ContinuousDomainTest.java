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

public class ContinuousDomainTest extends DomainTest {

	@Test
	public void fitTransform(){
		StructType schema = new StructType()
			.add("width", DataTypes.DoubleType, true)
			.add("height", DataTypes.DoubleType, true);

		List<Row> rows = Arrays.asList(
			RowFactory.create(20d, 10d),
			RowFactory.create(null, 20d),
			RowFactory.create(-999d, null),
			RowFactory.create(10d, Double.NaN),
			RowFactory.create(150d, 50d)
		);

		Dataset<Row> ds = SparkMLTest.sparkSession.createDataFrame(rows, schema);

		Map<String, Number[]> dataRanges = Collections.emptyMap();

		Map<String, List<Object>> expectedColumns;

		ContinuousDomain domain = new ContinuousDomain()
			.setInputCols(new String[]{"width", "height"})
			.setOutputCols(new String[]{"width_pmml", "height_pmml"});

		domain
			.setWithData(false);

		assertEquals("asIs", domain.getOutlierTreatment());
		assertEquals("asIs", domain.getMissingValueTreatment());
		assertEquals("returnInvalid", domain.getInvalidValueTreatment());

		ContinuousDomainModel domainModel = domain.fit(ds);

		Dataset<Row> transformedDs = domainModel.transform(ds);

		checkDataRanges(dataRanges, DomainUtil.toJavaMap(domainModel.getDataRanges()));

		dataRanges = Map.of(
			"width", new Number[]{-999d, 150d},
			"height", new Number[]{10d, 50d}
		);

		domain
			.setWithData(true);

		domainModel = domain.fit(ds);

		transformedDs = domainModel.transform(ds);

		checkDataRanges(dataRanges, DomainUtil.toJavaMap(domainModel.getDataRanges()));

		dataRanges = Map.of(
			"width", new Number[]{0d, 100d},
			"height", new Number[]{0d, 100d}
		);

		expectedColumns = Map.of(
			"width_pmml", Arrays.asList(20d, null, -1d, 10d, -1d),
			"height_pmml", Arrays.asList(10d, 20d, null, -1d, 50d)
		);

		domain
			.setDataRanges(DomainUtil.toScalaMap(dataRanges))
			.setInvalidValueTreatment("asValue")
			.setInvalidValueReplacement(-1d);

		domainModel = domain.fit(ds);

		transformedDs = domainModel.transform(ds);

		checkDataRanges(dataRanges, DomainUtil.toJavaMap(domainModel.getDataRanges()));

		checkDataset(expectedColumns, transformedDs);

		expectedColumns = Map.of(
			"width_pmml", Arrays.asList(20d, null, -1d, null, -1d),
			"height_pmml", Arrays.asList(null, 20d, null, -1d, 50d)
		);

		domain
			.setOutlierTreatment("asMissingValues")
			.setLowValue(20d)
			.setHighValue(100d);

		domainModel = domain.fit(ds);

		transformedDs = domainModel.transform(ds);

		checkDataset(expectedColumns, transformedDs);

		expectedColumns = Map.of(
			"width_pmml", Arrays.asList(20d, null, -1d, 20d, -1d),
			"height_pmml", Arrays.asList(20d, 20d, null, -1d, 50d)
		);

		domain
			.setOutlierTreatment("asExtremeValues");

		domainModel = domain.fit(ds);

		transformedDs = domainModel.transform(ds);

		checkDataset(expectedColumns, transformedDs);
	}

	static
	private void checkDataRanges(Map<String, Number[]> expected, Map<String, Number[]> actual){
		Set<String> keys = expected.keySet();

		assertEquals(keys, actual.keySet());

		for(String key : keys){
			List<Number> expectedValues = Arrays.asList(expected.get(key));
			List<Number> actualValues = Arrays.asList(actual.get(key));

			assertEquals(expectedValues, actualValues);
		}
	}
}