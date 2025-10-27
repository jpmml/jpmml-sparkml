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

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, Params, ParamValidators}
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCols}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{Column, Dataset, DataFrame}
import org.apache.spark.sql.functions.{col, lit, not, when}
import org.apache.spark.sql.types.{StructField, StructType}

sealed
trait MissingValueTreatment {
	def name: String
}

object MissingValueTreatment {
	case object AsIs extends MissingValueTreatment { 
		val name = "asIs"
	}

	case object AsMean extends MissingValueTreatment {
		val name = "asMean"
	}

	case object AsMode extends MissingValueTreatment {
		val name = "asMode" 
	}

	case object AsMedian extends MissingValueTreatment {
		val name = "asMedian"
	}

	case object AsValue extends MissingValueTreatment {
		val name = "asValue"
	}

	case object ReturnInvalid extends MissingValueTreatment {
		val name = "returnInvalid"
	}

	val values: Array[MissingValueTreatment] = Array(AsIs, AsMean, AsMode, AsMedian, AsValue, ReturnInvalid)

	def forName(name: String): MissingValueTreatment = values
		.find(_.name == name)
		.getOrElse(throw new IllegalArgumentException(s"${name}")
	)
}

sealed
trait InvalidValueTreatment {
	def name: String
}

object InvalidValueTreatment {
	case object ReturnInvalid extends InvalidValueTreatment {
		val name = "returnInvalid"
	}

	case object AsIs extends InvalidValueTreatment {
		val name = "asIs"
	}

	case object AsMissing extends InvalidValueTreatment {
		val name = "asMissing"
	}

	case object AsValue extends InvalidValueTreatment {
		val name = "asValue"
	}

	val values: Array[InvalidValueTreatment] = Array(ReturnInvalid, AsIs, AsMissing, AsValue)

	def forName(name: String): InvalidValueTreatment = values
		.find(_.name == name)
		.getOrElse(throw new IllegalArgumentException(s"${name}")
	)
}

trait HasDomainParams[T <: HasDomainParams[T]] extends Params with HasInputCols with HasOutputCols {

	val missingValues: Param[Array[Object]] = new Param[Array[Object]](this, "missingValues", "")

	val missingValueTreatment: Param[String] = new Param[String](this, "missingValueTreatment", "", ParamValidators.inArray(MissingValueTreatment.values.map(_.name)))

	val missingValueReplacement: Param[Object] = new Param[Object](this, "missingValueReplacement", "")

	val invalidValueTreatment: Param[String] = new Param[String](this, "invalidValueTreatment", "", ParamValidators.inArray(InvalidValueTreatment.values.map(_.name)))

	val invalidValueReplacement: Param[Object] = new Param[Object](this, "invalidValueReplacement", "")

	val withData: BooleanParam = new BooleanParam(this, "withData", "")

	private 
	lazy 
	val missingValuesSet: Set[Object] = if (isDefined(missingValues)) getMissingValues.toSet else Set.empty

	protected
	def self: T = this.asInstanceOf[T]

	def setInputCols(value: Array[String]): T = {
		set(inputCols, value)
		self
	}

	def setOutputCols(value: Array[String]): T = {
		set(outputCols, value)
		self
	}

	def getMissingValues: Array[Object] = $(missingValues)

	def setMissingValues(value: Array[Object]): T = {
		set(missingValues, value)
		self
	}

	def getMissingValueTreatment: String = $(missingValueTreatment)

	def setMissingValueTreatment(value: String): T = {
		set(missingValueTreatment, value)
		self
	}

	def getMissingValueReplacement: Object = $(missingValueReplacement)

	def setMissingValueReplacement(value: Object): T = {
		set(missingValueReplacement, value)
		self
	}

	def getInvalidValueTreatment: String = $(invalidValueTreatment)

	def setInvalidValueTreatment(value: String): T = {
		set(invalidValueTreatment, value)
		self
	}

	def getInvalidValueReplacement: Object = $(invalidValueReplacement)

	def setInvalidValueReplacement(value: Object): T = {
		set(invalidValueReplacement, value)
		self
	}

	def getWithData: Boolean = $(withData)

	def setWithData(value: Boolean): T = {
		set(withData, value)
		self
	}


	protected
	def isMissing(col: Column): Column = {

		if(missingValuesSet.nonEmpty){
			col.isin(missingValuesSet.toSeq: _*)
		} else

		{
			col.isNull
		}
	}

	protected
	def isValid(col: Column, isNotMissingCol: Column): Column = {
		isNotMissingCol
	}

	protected
	def isInvalid(col: Column, isMissingCol: Column, isValidCol: Column): Column = {
		not(isMissingCol) && not(isValidCol)
	}

	protected 
	def validateParams(): Unit = {
		require(isDefined(inputCols) && isDefined(outputCols), "inputCols and outputCols must be defined")
		require(getInputCols.length == getOutputCols.length, "inputCols and outputCols must have the same length")

		if(isDefined(missingValueReplacement)){

			if(getMissingValueTreatment == MissingValueTreatment.ReturnInvalid.name){
				throw new IllegalArgumentException(s"Missing value treatment ${getMissingValueTreatment} does not support missingValueReplacement")
			}
		}

		if(isDefined(invalidValueReplacement)){

			if(getInvalidValueTreatment != InvalidValueTreatment.AsValue.name){
				throw new IllegalArgumentException(s"Invalid value treatment ${getInvalidValueTreatment} does not support invalidValueReplacement")
			}
		} else

		{
			if(getInvalidValueTreatment == InvalidValueTreatment.AsValue.name){
				throw new IllegalArgumentException(s"Invalid value treatment ${getInvalidValueTreatment} requires invalidValueReplacement")
			}
		}
	}

	def transformSchema(schema: StructType): StructType = {
		validateParams()

		val inputColNames = getInputCols
		val outputColNames = getOutputCols

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

		StructField(outputColName, inputField.dataType, nullable = true)
	}
}

abstract
class Domain[E <: Domain[E, M], M <: DomainModel[M]](override val uid: String) extends Estimator[M] with HasDomainParams[E] with DefaultParamsWritable {

	setDefault(
		missingValues -> Array.empty[Object],
		missingValueTreatment -> MissingValueTreatment.AsIs.name,
		invalidValueTreatment -> InvalidValueTreatment.ReturnInvalid.name,
		withData -> true
	)

	override
	def copy(extra: ParamMap): Domain[E, M] = defaultCopy(extra)

	protected
	def selectNonMissing(inputColNames: Array[String]): Seq[Column] = {
		inputColNames.map {
			inputColName => {
				val inputCol = col(inputColName)

				val isMissingCol = isMissing(inputCol)

				val isNotMissingCol = not(isMissingCol)

				when(isNotMissingCol, inputCol).as(inputColName)
			}
		}
	}
}

abstract
class DomainModel[M <: DomainModel[M]](override val uid: String) extends Model[M] with HasDomainParams[M] with DefaultParamsWritable {

	protected
	def transformMissing(col: Column, isMissingCol: Column): Column = {
		val missingValueTreatment = getMissingValueTreatment
		val missingValueReplacement = getMissingValueReplacement

		MissingValueTreatment.forName(missingValueTreatment) match {
			case MissingValueTreatment.AsIs => 
				col
			case MissingValueTreatment.AsMean |
				MissingValueTreatment.AsMode |
				MissingValueTreatment.AsMedian |
				MissingValueTreatment.AsValue =>
				when(isMissingCol, lit(missingValueReplacement)).otherwise(col)
			case _ => 
				throw new IllegalArgumentException(missingValueTreatment)
		}
	}

	protected
	def transformValid(col: Column, isValidCol: Column): Column = {
		col
	}

	protected
	def transformInvalid(col: Column, isInvalidCol: Column): Column = {
		val invalidValueTreatment = getInvalidValueTreatment
		val invalidValueReplacement = getInvalidValueReplacement

		InvalidValueTreatment.forName(invalidValueTreatment) match {
			case InvalidValueTreatment.AsIs =>
				col
			case InvalidValueTreatment.AsMissing => {
				val nullifiedCol = when(isInvalidCol, lit(null)).otherwise(col)
				transformMissing(nullifiedCol, isInvalidCol)
			}
			case InvalidValueTreatment.AsValue =>
				when(isInvalidCol, lit(invalidValueReplacement)).otherwise(col)
			case _ =>
				throw new IllegalArgumentException(invalidValueTreatment)
		}
	}

	override
	def transform(dataset: Dataset[_]): DataFrame = {
		val inputColNames = getInputCols
		val outputColNames = getOutputCols

		val outputCols = inputColNames.zip(outputColNames).map {
			case (inputColName, outputColName) => {
				var outputCol = col(inputColName).as(outputColName)

				val isMissingCol = isMissing(outputCol)
				outputCol = transformMissing(outputCol, isMissingCol)

				val isNotMissingCol = not(isMissingCol)

				val isValidCol = isValid(outputCol, isNotMissingCol)
				outputCol = transformValid(outputCol, isValidCol)

				val isInvalidCol = isInvalid(outputCol, isMissingCol, isValidCol)
				outputCol = transformInvalid(outputCol, isInvalidCol)

				outputCol
			}
		}

		dataset.select(dataset("*") +: outputCols: _*)
	}
}
