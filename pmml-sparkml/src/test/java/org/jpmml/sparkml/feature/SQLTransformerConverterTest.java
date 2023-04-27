/*
 * Copyright (c) 2021 Villu Ruusmann
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
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.jpmml.sparkml.ConverterFactory;
import org.jpmml.sparkml.DatasetUtil;
import org.jpmml.sparkml.SparkMLEncoder;
import org.jpmml.sparkml.SparkSessionUtil;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class SQLTransformerConverterTest {

	@Test
	public void encodeLogicalPlan(){
		List<String> dataFieldNames = Arrays.asList("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Species");
		List<String> derivedFieldNames = Arrays.asList();

		checkFields("SELECT * FROM __THIS__", dataFieldNames, derivedFieldNames);
		checkFields("SELECT * FROM (SELECT * FROM __THIS__)", dataFieldNames, derivedFieldNames);

		derivedFieldNames = Arrays.asList("SL", "SW", "PL", "PW");

		checkFields("SELECT * FROM (SELECT Sepal_Length AS SL, Sepal_Width AS SW, Petal_Length AS PL, Petal_Width AS PW FROM __THIS__)", dataFieldNames, derivedFieldNames);
		checkFields("SELECT * FROM (SELECT *, Sepal_Length AS SL, Sepal_Width AS SW, Petal_Length AS PL, Petal_Width AS PW FROM __THIS__)", dataFieldNames, derivedFieldNames);

		derivedFieldNames = Arrays.asList("Length_Ratio", "Width_Ratio");

		checkFields("SELECT * FROM (SELECT *, Sepal_Length / Petal_Length AS Length_Ratio, Sepal_Width / Petal_Width AS Width_Ratio FROM __THIS__)", dataFieldNames, derivedFieldNames);

		derivedFieldNames = Arrays.asList("SL", "SW", "PL", "PW", "Length_Ratio", "Width_Ratio");

		checkFields("SELECT *, SL / PL AS Length_Ratio, SW / PW AS Width_Ratio FROM (SELECT *, Sepal_Length AS SL, Sepal_Width AS SW, Petal_Length AS PL, Petal_Width AS PW FROM __THIS__)", dataFieldNames, derivedFieldNames);
	}

	static
	private void checkFields(String sqlStatement, Collection<String> dataFieldNames, Collection<String> derivedFieldNames){
		dataFieldNames = new LinkedHashSet<>(dataFieldNames);
		derivedFieldNames = new LinkedHashSet<>(derivedFieldNames);

		ConverterFactory converterFactory = new ConverterFactory(Collections.emptyMap());

		SparkMLEncoder encoder = new SparkMLEncoder(SQLTransformerConverterTest.schema, converterFactory);

		LogicalPlan logicalPlan = DatasetUtil.createAnalyzedLogicalPlan(SQLTransformerConverterTest.sparkSession, SQLTransformerConverterTest.schema, sqlStatement);

		SQLTransformerConverter.encodeLogicalPlan(encoder, logicalPlan);

		Collection<DataField> dataFields = (encoder.getDataFields()).values();
		for(DataField dataField : dataFields){
			String name = dataField.requireName();

			assertTrue(name, dataFieldNames.remove(name));
		}

		assertTrue(dataFieldNames.toString(), dataFieldNames.isEmpty());

		Collection<DerivedField> derivedFields = (encoder.getDerivedFields()).values();
		for(DerivedField derivedField : derivedFields){
			String name = derivedField.requireName();

			assertTrue(name, derivedFieldNames.remove(name));
		}

		assertTrue(derivedFieldNames.toString(), derivedFieldNames.isEmpty());
	}

	@BeforeClass
	static
	public void createSparkSession(){
		SQLTransformerConverterTest.sparkSession = SparkSessionUtil.createSparkSession();
	}

	@AfterClass
	static
	public void destroySparkSession(){
		SQLTransformerConverterTest.sparkSession = SparkSessionUtil.destroySparkSession(SQLTransformerConverterTest.sparkSession);
	}

	public static SparkSession sparkSession = null;

	private static final StructType schema = new StructType()
		.add("Sepal_Length", DataTypes.DoubleType)
		.add("Sepal_Width", DataTypes.DoubleType)
		.add("Petal_Length", DataTypes.DoubleType)
		.add("Petal_Width", DataTypes.DoubleType)
		.add("Species", DataTypes.StringType);
}