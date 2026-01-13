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
package org.jpmml.sparkml.feature

import java.util.Arrays

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCols}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DoubleType, StructType, StructField}

class VectorDisassembler(override val uid: String) extends Transformer with HasInputCol with HasOutputCols with DefaultParamsWritable {

	private val extractElement = udf((vector: Vector, index: Int) => {
		(vector match {
			case sparseVec: SparseVector => {
				val pos = Arrays.binarySearch(sparseVec.indices, index)
				if(pos >= 0){
					Some(sparseVec.values(pos))
				} else

				{
					None
				}
			}
			case denseVec: DenseVector => {
				Some(denseVec(index))
			}
		}) : Option[Double]
	})

	/**
	 * @group setParam
	 */
	def setInputCol(value: String): this.type = set(inputCol, value)

	/**
	 * @group setParam
	 */
	def setOutputCols(value: Array[String]): this.type = set(outputCols, value)


	def this() = this(Identifiable.randomUID("vecDisassembler"))

	override
	def copy(extra: ParamMap): VectorDisassembler = defaultCopy(extra)

	protected def validateParams(): Unit = {
		require(isDefined(inputCol) && isDefined(outputCols), "inputCol and outputCols must be defined")
	}

	override
	def transformSchema(schema: StructType): StructType = {
		validateParams()

		val inputColName = getInputCol
		val outputColNames = getOutputCols

		val inputFields = schema.fields

		require(inputFields.exists(_.name == inputColName), s"Input column $inputColName not found")
		outputColNames.foreach {
			outputColName => require(!inputFields.exists(_.name == outputColName), s"Output column $outputColName already exists")
		}

		val inputField = schema(inputColName)
		require(inputField.dataType.typeName == "vector", s"Input column $inputColName must be of type VectorUDT, found ${inputField.dataType}")

		val outputFields = outputColNames.map {
			outputColName => StructField(outputColName, DoubleType, nullable = true)
		}

		StructType(inputFields ++ outputFields)
	}

	override 
	def transform(dataset: Dataset[_]): DataFrame = {
		val inputColName = getInputCol
		val outputColNames = getOutputCols

		val firstVector = dataset.select(col(inputColName))
			.head()
			.getAs[Vector](0)

		require(outputColNames.length == firstVector.size, s"Number of output columns must equal vector size")

		val df = dataset.toDF()

		outputColNames.indices.foldLeft(df){
			(df, i) => df.withColumn(outputColNames(i), extractElement(col(inputColName), lit(i)))
		}
	}
}

object VectorDisassembler extends DefaultParamsReadable[VectorDisassembler]