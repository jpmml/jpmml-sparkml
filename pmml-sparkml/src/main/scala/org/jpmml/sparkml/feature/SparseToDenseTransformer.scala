/*
 * Copyright (c) 2020 Villu Ruusmann
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
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StructField, StructType}

class SparseToDenseTransformer(override val uid: String) extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

	def this() = this(Identifiable.randomUID("sparse2dense"))

	def setInputCol(value: String): this.type = set(inputCol, value)

	def setOutputCol(value: String): this.type = set(outputCol, value)

	override
	def copy(extra: ParamMap): SparseToDenseTransformer = defaultCopy(extra)

	override
	def transformSchema(schema: StructType): StructType = {
		val inputColName = $(inputCol)
		val outputColName = $(outputCol)

		val inputFields = schema.fields

		require(!inputFields.exists(_.name == outputColName), s"Output column $outputColName already exists")

		val inputField = schema(inputColName)
		val outputField = new StructField(outputColName, inputField.dataType, inputField.nullable)

		StructType(inputFields :+ outputField)
	}

	override 
	def transform(dataset: Dataset[_]): Dataset[Row] = {
		val inputColName = $(inputCol)
		val outputColName = $(outputCol)

		transformSchema(dataset.schema, logging = true)

		val converter = udf { vec: Vector => vec.toDense }

		dataset.withColumn(outputColName, converter(dataset(inputColName)))
	}
}

object SparseToDenseTransformer extends DefaultParamsReadable[SparseToDenseTransformer] {

	override
	def load(path: String): SparseToDenseTransformer = super.load(path)
}
