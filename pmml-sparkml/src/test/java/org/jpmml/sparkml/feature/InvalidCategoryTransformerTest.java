/*
 * Copyright (c) 2023 Villu Ruusmann
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
import java.util.List;
import java.util.Objects;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.attribute.Attribute;
import org.apache.spark.ml.attribute.AttributeKeys;
import org.apache.spark.ml.attribute.AttributeType;
import org.apache.spark.ml.attribute.NominalAttribute;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.jpmml.sparkml.SparkMLTest;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class InvalidCategoryTransformerTest extends SparkMLTest {

	@Test
	public void transform(){
		StructType schema = new StructType()
			.add("fruit", DataTypes.StringType, true)
			.add("color", DataTypes.StringType, true)
			.add("rating", DataTypes.DoubleType, false);

		List<Row> rows = Arrays.asList(
			RowFactory.create("apple", "red", 2d),
			RowFactory.create("orange", "orange", 3d),
			RowFactory.create("banana", "yellow", 3d),
			RowFactory.create("banana", "green", 1d),
			RowFactory.create("apple", "green", 2d)
		);

		Dataset<Row> ds = SparkMLTest.sparkSession.createDataFrame(rows, schema);

		List<PipelineStage> stages = new ArrayList<>();

		StringIndexer stringIndexer = new StringIndexer()
			.setStringOrderType("alphabetAsc")
			.setInputCols(new String[]{"fruit", "color", "rating"})
			.setOutputCols(new String[]{"fruitIdx", "colorIdx", "ratingIdx"})
			.setHandleInvalid("keep");

		stages.add(stringIndexer);

		String[] indexedCols = stringIndexer.getOutputCols();
		for(String indexedCol : indexedCols){
			InvalidCategoryTransformer invalidCategoryTransformer = new InvalidCategoryTransformer()
				.setInputCol(indexedCol)
				.setOutputCol(indexedCol + "Transformed");

			stages.add(invalidCategoryTransformer);
		}

		Pipeline pipeline = new Pipeline()
			.setStages(stages.toArray(new PipelineStage[stages.size()]));

		PipelineModel pipelineModel = pipeline.fit(ds);

		Dataset<Row> transformedDs = pipelineModel.transform(ds);

		StructType transformedSchema = transformedDs.schema();

		NominalAttribute fruitIdxAttr = (NominalAttribute)getAttribute(transformedSchema, "fruitIdx");
		NominalAttribute colorIdxAttr = (NominalAttribute)getAttribute(transformedSchema, "colorIdx");
		NominalAttribute ratingIdxAttr = (NominalAttribute)getAttribute(transformedSchema, "ratingIdx");

		assertArrayEquals(new String[]{"apple", "banana", "orange", "__unknown"}, (fruitIdxAttr.values()).get());
		assertArrayEquals(new String[]{"green", "orange", "red", "yellow", "__unknown"}, (colorIdxAttr.values()).get());
		assertArrayEquals(new String[]{"1.0", "2.0", "3.0", "__unknown"}, (ratingIdxAttr.values()).get());

		NominalAttribute fruitIdxTransformedAttr = (NominalAttribute)getAttribute(transformedSchema, "fruitIdxTransformed");
		NominalAttribute colorIdxTransformedAttr = (NominalAttribute)getAttribute(transformedSchema, "colorIdxTransformed");
		NominalAttribute ratingIdxTransformedAttr = (NominalAttribute)getAttribute(transformedSchema, "ratingIdxTransformed");

		assertArrayEquals(new String[]{"apple", "banana", "orange"}, (fruitIdxTransformedAttr.values()).get());
		assertArrayEquals(new String[]{"green", "orange", "red", "yellow"}, (colorIdxTransformedAttr.values()).get());
		assertArrayEquals(new String[]{"1.0", "2.0", "3.0"}, (ratingIdxTransformedAttr.values()).get());

		List<Row> testRows = Arrays.asList(
			RowFactory.create(null, "yellow", 0d),
			RowFactory.create("apple", "", 1d),
			RowFactory.create("banana", "red", Double.NaN)
		);

		Dataset<Row> testDs = SparkMLTest.sparkSession.createDataFrame(testRows, schema);

		Dataset<Row> transformedTestDs = pipelineModel.transform(testDs);

		List<Row> transformedTestRows = transformedTestDs
			.select("fruitIdxTransformed", "colorIdxTransformed", "ratingIdxTransformed")
			.collectAsList();

		assertEquals(3, transformedTestRows.size());

		assertArrayEquals(new Object[]{Double.NaN, 3d, Double.NaN}, getValues(transformedTestRows.get(0)));
		assertArrayEquals(new Object[]{0d, Double.NaN, 0d}, getValues(transformedTestRows.get(1)));
		assertArrayEquals(new Object[]{1d, 2d, Double.NaN}, getValues(transformedTestRows.get(2)));
	}

	static
	private Attribute getAttribute(StructType schema, String name){
		StructField structField = schema.apply(name);

		Metadata metadata = structField.metadata();

		String mlAttr = AttributeKeys.ML_ATTR();

		if(metadata.contains(mlAttr)){
			Metadata mlAttrMetadata = metadata.getMetadata(mlAttr);

			if(!Objects.equals(mlAttrMetadata.getString("type"), AttributeType.Nominal().name())){
				throw new IllegalArgumentException();
			}

			return NominalAttribute.fromStructField(structField);
		}

		return null;
	}

	static
	private Object[] getValues(Row row){
		Object[] result = new Object[row.size()];

		for(int i = 0, max = row.size(); i < max; i++){
			result[i] = row.get(i);
		}

		return result;
	}
}