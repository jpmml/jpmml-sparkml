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
import org.apache.spark.ml.param.{ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasInputCol, HasInputCols, HasOutputCol, HasOutputCols}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{Column, Dataset, Row}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StructField, StructType}

class InvalidCategoryTransformer(override val uid: String) extends Transformer with HasInputCol with HasInputCols with HasOutputCol with HasOutputCols with DefaultParamsWritable {

	def this() = this(Identifiable.randomUID("invalidCat"))

	def setInputCol(value: String): this.type = set(inputCol, value)

	def setOutputCol(value: String): this.type = set(outputCol, value)

	def setInputCols(value: Array[String]): this.type = set(inputCols, value)

	def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

	override
	def copy(extra: ParamMap): InvalidCategoryTransformer = defaultCopy(extra)

	override
	def transformSchema(schema: StructType): StructType = {
		val (inputColNames, outputColNames) = getInOutCols()

		val inputFields = schema.fields
		val outputFields = new Array[StructField](outputColNames.length)

		for(i <- 0 until outputColNames.length){
			val inputColName = inputColNames(i)
			val outputColName = outputColNames(i)

			require(!inputFields.exists(_.name == outputColName), s"Output column $outputColName already exists")

			outputFields(i) = transformField(schema, inputColName, outputColName)
		}

		StructType(inputFields ++ outputFields)
	}

	override 
	def transform(dataset: Dataset[_]): Dataset[Row] = {
		val (inputColNames, outputColNames) = getInOutCols()

		val transformedSchema = transformSchema(dataset.schema, logging = true)

		var result = dataset.asInstanceOf[Dataset[Row]]

		for(i <- 0 until outputColNames.length){
			val inputColName = inputColNames(i)
			val outputColName = outputColNames(i)

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

			val outputColumn = converter(dataset(inputColName)).as(outputColName, outputMlAttr)

			result = result.withColumn(outputColName, outputColumn)
		}

		result
	}

	private def getInOutCols(): (Array[String], Array[String]) = {
		ParamValidators.checkSingleVsMultiColumnParams(this, Seq(outputCol), Seq(outputCols))

		if(isSet(inputCol)){
			(Array($(inputCol)), Array($(outputCol)))
		} else 

		{
			require($(inputCols).length == $(outputCols).length)

			($(inputCols), $(outputCols))
		}
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

object InvalidCategoryTransformer extends DefaultParamsReadable[InvalidCategoryTransformer] {

	override
	def load(path: String): InvalidCategoryTransformer = super.load(path)
}
