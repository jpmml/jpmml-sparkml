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
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

class InvalidCategoryTransformer(override val uid: String) extends Transformer with HasInputCol with HasInputCols with HasOutputCol with HasOutputCols with DefaultParamsWritable {

	private
	val convertUDF = udf {
		(x: Double, size: Int) => {
			if(x >= 0 && x < size){
				x
			} else

			{
				Double.NaN
			}
		}
	}

	def setInputCol(value: String): this.type = set(inputCol, value)

	def setOutputCol(value: String): this.type = set(outputCol, value)

	def setInputCols(value: Array[String]): this.type = set(inputCols, value)

	def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

	protected 
	def getInOutCols(): (Array[String], Array[String]) = {

		if(isSet(inputCol)){
			(Array(getInputCol), Array(getOutputCol))
		} else 

		{
			(getInputCols, getOutputCols)
		}
	}


	def this() = this(Identifiable.randomUID("invalidCat"))

	override
	def copy(extra: ParamMap): InvalidCategoryTransformer = defaultCopy(extra)

	protected 
	def validateParams(): Unit = {
		ParamValidators.checkSingleVsMultiColumnParams(this, Seq(inputCol, outputCol), Seq(inputCols, outputCols))

		if(isSet(inputCol)){
			require(isDefined(inputCol) && isDefined(outputCol), "inputCol and outputCol must be defined")
		} else

		{
			require(isDefined(inputCols) && isDefined(outputCols), "inputCols and outputCols must be defined")
			require(getInputCols.length == getOutputCols.length, "inputCols and outputCols must have the same length")
		}
	}

	override
	def transformSchema(schema: StructType): StructType = {
		validateParams()

		val (inputColNames, outputColNames) = getInOutCols()

		val inputFields = schema.fields

		val outputFields = inputColNames.zip(outputColNames).map {
			case (inputColName, outputColName) => {
				require(inputFields.exists(_.name == inputColName), s"Input column ${inputColName} not found")
				require(!inputFields.exists(_.name == outputColName), s"Output column ${outputColName} already exists")
			
				transformField(schema, inputColName, outputColName)
			}
		}

		StructType(inputFields ++ outputFields)
	}

	private
	def transformField(schema: StructType, inputColName: String, outputColName: String): StructField = {
		val inputField = schema(inputColName)

		require(inputField.dataType == DoubleType, s"Input column ${inputColName} must be of type DoubleType, found ${inputField.dataType}")

		val inputMlAttr = NominalAttribute.fromStructField(inputField).asInstanceOf[NominalAttribute]
		// XXX
		//require(inputMlAttr.values.isDefined && inputMlAttr.values.get.last == "__unknown", s"Input column ${inputColName} must have StringIndexer-like NominalAttribute metadata")

		var outputMlAttr = NominalAttribute.defaultAttr
			.withName(outputColName)
			.toMetadata()

		StructField(outputColName, inputField.dataType, inputField.nullable, outputMlAttr)
	}

	override 
	def transform(dataset: Dataset[_]): Dataset[Row] = {
		val transformedSchema = transformSchema(dataset.schema, logging = true)

		val (inputColNames, outputColNames) = getInOutCols()

		val outputCols = inputColNames.zip(outputColNames).map {
			case (inputColName, outputColName) => {
				val inputField = transformedSchema(inputColName)
				val inputMlAttr = NominalAttribute.fromStructField(inputField).asInstanceOf[NominalAttribute]
				val inputLabels = inputMlAttr.values.get
				require(inputLabels.last == "__unknown")

				val outputLabels = inputLabels.slice(0, inputLabels.size - 1)
				val outputMlAttr = NominalAttribute.defaultAttr
					.withName(outputColName)
					.withValues(outputLabels.asInstanceOf[Array[String]])
					.toMetadata()

				convertUDF(dataset(inputColName), lit(outputLabels.length)).as(outputColName, outputMlAttr)
			}
		}

		dataset.select(dataset("*") +: outputCols: _*)
	}
}

object InvalidCategoryTransformer extends DefaultParamsReadable[InvalidCategoryTransformer]
