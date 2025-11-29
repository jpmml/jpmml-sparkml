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
package org.jpmml.sparkml;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldUtil;
import org.jpmml.converter.ObjectFeature;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

public class SparkMLEncoderTest {

	@Test
	public void toCategorical(){
		StructType schema = new StructType()
			.add("x", DataTypes.IntegerType, false);

		ConverterFactory converterFactory = new ConverterFactory(Collections.emptyMap());

		SparkMLEncoder encoder = new SparkMLEncoder(schema, converterFactory);

		Feature feature = encoder.getOnlyFeature("x");

		assertInstanceOf(ContinuousFeature.class, feature);

		DataField dataField = checkField(feature, OpType.CONTINUOUS, DataType.INTEGER, Collections.emptyList());

		encoder.toCategorical(feature, Arrays.asList(1, 2, 3));

		// Clear feature cache
		encoder.removeFeatures("x");

		feature = encoder.getOnlyFeature("x");

		assertInstanceOf(ObjectFeature.class, feature);

		dataField = checkField(feature, OpType.CATEGORICAL, DataType.INTEGER, Arrays.asList(1, 2, 3));

		encoder.toCategorical(feature, Arrays.asList("1.0", "2.0", "3.0"));
	}

	static
	private DataField checkField(Feature feature, OpType opType, DataType dataType, List<?> values){
		DataField field = (DataField)feature.getField();

		assertEquals(opType, field.requireOpType());
		assertEquals(dataType, field.requireDataType());
		assertEquals(values, FieldUtil.getValues(field));

		return field;
	}
}