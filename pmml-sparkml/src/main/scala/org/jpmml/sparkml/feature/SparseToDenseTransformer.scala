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
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StructField, StructType}

class SparseToDenseTransformer(override val uid: String) extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {
	private
	val sparseToDenseUDF = udf(SparseToDenseTransformer.sparseToDense _)

	/**
	 * @group setParam
	 */
	def setInputCol(value: String): this.type = set(inputCol, value)

	/**
	 * @group setParam
	 */
	def setOutputCol(value: String): this.type = set(outputCol, value)


	def this() = this(Identifiable.randomUID("sparse2dense"))

	override
	def copy(extra: ParamMap): SparseToDenseTransformer = defaultCopy(extra)

	protected 
	def validateParams(): Unit = {
		require(isDefined(inputCol) && isDefined(outputCol), "inputCol and outputCol must be defined")
	}

	override
	def transformSchema(schema: StructType): StructType = {
		validateParams()

		val inputColName = getInputCol
		val outputColName = getOutputCol

		val inputFields = schema.fields

		require(inputFields.exists(_.name == inputColName), s"Input column $inputColName not found")
		require(!inputFields.exists(_.name == outputColName), s"Output column $outputColName already exists")

		val inputField = schema(inputColName)
		val outputField = new StructField(outputColName, inputField.dataType, inputField.nullable)

		StructType(inputFields :+ outputField)
	}

	override 
	def transform(dataset: Dataset[_]): Dataset[Row] = {
		val inputColName = getInputCol
		val outputColName = getOutputCol

		dataset.withColumn(outputColName, sparseToDenseUDF(dataset(inputColName)))
	}
}

object SparseToDenseTransformer extends DefaultParamsReadable[SparseToDenseTransformer] {

	def sparseToDense(vec: Vector): DenseVector = {
		if(vec != null){
			vec match {
				case denseVec: DenseVector => denseVec
				case _ => vec.toDense
			}
		} else

		{
			null
		}
	}
}
