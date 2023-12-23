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
package org.jpmml.sparkml.feature

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StructField, StructType}

class InvalidCategoryTransformer(override val uid: String) extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

	def this() = this(Identifiable.randomUID("invalidCat"))

	def setInputCol(value: String): this.type = set(inputCol, value)

	def setOutputCol(value: String): this.type = set(outputCol, value)

	override
	def copy(extra: ParamMap): InvalidCategoryTransformer = defaultCopy(extra)

	override
	def transformSchema(schema: StructType): StructType = {
		val inputColName = $(inputCol)
		val outputColName = $(outputCol)

		val inputFields = schema.fields

		require(!inputFields.exists(_.name == outputColName), s"Output column $outputColName already exists")

		val outputField = transformField(schema, inputColName, outputColName)

		StructType(inputFields :+ outputField)
	}

	override 
	def transform(dataset: Dataset[_]): Dataset[Row] = {
		val inputColName = $(inputCol)
		val outputColName = $(outputCol)

		val transformedSchema = transformSchema(dataset.schema, logging = true)

		val inputField = transformedSchema(inputColName)
		val outputField = transformedSchema(outputColName)

		val inputMlAttr = NominalAttribute.fromStructField(inputField).asInstanceOf[NominalAttribute]
		require(inputMlAttr.values.isDefined)

		val inputLabels = inputMlAttr.values.get
		require(inputLabels.last == "__unknown")

		val outputLabels = inputLabels.slice(0, inputLabels.size - 1)
		val outputMlAttr = NominalAttribute.defaultAttr
			.withName(outputColName)
			.withValues(outputLabels.asInstanceOf[Array[String]])
			.toMetadata()

		val converter = udf { x: Double => if (x >= 0 && x < outputLabels.size) x else Double.NaN }

		dataset.withColumn(outputColName, converter(dataset(inputColName)).as(outputColName, outputMlAttr))
	}

	private def transformField(schema: StructType, inputColName: String, outputColName: String): StructField = {
		val inputField = schema(inputColName)

		val inputMlAttr = NominalAttribute.fromStructField(inputField).asInstanceOf[NominalAttribute]

		var outputMlAttr = NominalAttribute.defaultAttr
			.withName(outputColName)
			.toMetadata()

		StructField(outputColName, inputField.dataType, inputField.nullable, outputMlAttr)
	}
}