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
import org.apache.spark.sql.functions.{col, lit, not, raise_error, when}
import org.apache.spark.sql.types.{StructField, StructType}

/**
 * Strategy for handling missing values during transformation.
 *
 * <p>
 * The default representation of missing values is <code>null</code>.
 * Override by specifying a non-empty [[HasDomainParams.missingValues]].
 * </p>
 *
 * <h2>Supported strategies</h2>
 * 
 * <ul>
 *   <li><code>asIs</code>. Leave missing values unchanged.
 *   <li><code>asValue</code> (aliases: <code>asMean</code>, <code>asMode</code>, <code>asMean</code>). Replace missing values with [[HasDomainParams.missingValueReplacement]].
 *   <li><code>returnInvalid</code>. Raise error.
 * </ul>
 *
 * @see <a href="https://dmg.org/pmml/v4-4-1/MiningSchema.html#xsdType_MISSING-VALUE-TREATMENT-METHOD">PMML specification</a>
 */
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
		.getOrElse(throw new IllegalArgumentException(name))
}

/**
 * Strategy for handling invalid values during transformation.
 *
 * <p>
 * The default representation of invalid floating-point values is </code>NaN</code>. There is no default representation for integer values, or non-numeric values.
 * Override by specifying a non-empty [[HasCategoricalDomainParams.dataValues]] or [[HasContinuousDomainParams.dataRanges]].
 * </p>
 *
 * <h2>Supported strategies</h2>
 * 
 * <ul>
 *   <li><code>asIs</code>. Leave invalid values unchanged.
 *   <li><code>asValue</code>. Replace invalid values with [[HasDomainParams.invalidValueReplacement]].
 *   <li><code>asMissing</code> Replace invalid values with missing values, and apply the effective missing value treatment to them.
 *   <li><code>returnInvalid</code>. Raise error.
 * </ul>
 *
 * @see <a href="https://dmg.org/pmml/v4-4-1/MiningSchema.html#xsdType_INVALID-VALUE-TREATMENT-METHOD">PMML specification</a>
 */
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
		.getOrElse(throw new IllegalArgumentException(name))
}

trait HasDomainParams[T <: HasDomainParams[T]] extends Params with HasInputCols with HasOutputCols {

	/**
	 * @group param
	 */
	val missingValues: Param[Array[Object]] = new Param[Array[Object]](this, "missingValues", "Values that equate to missing values")

	/**
	 * Supported values: <code>asIs</code>, <code>asValue</code>, <code>returnInvalid</code>.
	 * Default: <code>asIs</code>.
	 *
	 * @see [[MissingValueTreatment]]
	 * @group param
	 */
	val missingValueTreatment: Param[String] = new Param[String](this, "missingValueTreatment", "Missing value handling strategy", ParamValidators.inArray(MissingValueTreatment.values.map(_.name)))

	/**
	 * @group param
	 */
	val missingValueReplacement: Param[Object] = new Param[Object](this, "missingValueReplacement", "Missing value replacement value")

	/**
	 * Supported values: <code>asIs</code>, <code>asValue</code>, <code>asMissing</code>, <code>returnInvalid</code>.
	 * Default: <code>returnInvalid</code>.
	 *
	 * @see [[InvalidValueTreatment]]
	 * @group param
	 */
	val invalidValueTreatment: Param[String] = new Param[String](this, "invalidValueTreatment", "Invalid value handling strategy", ParamValidators.inArray(InvalidValueTreatment.values.map(_.name)))

	/**
	 * @group param
	 */
	val invalidValueReplacement: Param[Object] = new Param[Object](this, "invalidValueReplacement", "Invalid value replacement value")

	/**
	 * Default: <code>true</code>.
	 * 
	 * @group param
	 */
	val withData: BooleanParam = new BooleanParam(this, "withData", "Collect valid value information during fitting?")


	protected
	def self: T = this.asInstanceOf[T]

	/**
	 * @group setParam
	 */
	def setInputCols(value: Array[String]): T = {
		set(inputCols, value)
		self
	}

	/**
	 * @group setParam
	 */
	def setOutputCols(value: Array[String]): T = {
		set(outputCols, value)
		self
	}

	/**
	 * @group getParam
	 */
	def getMissingValues: Array[Object] = $(missingValues)

	/**
	 * @group setParam
	 */
	def setMissingValues(value: Array[Object]): T = {
		set(missingValues, value)
		self
	}

	/**
	 * @group getParam
	 */
	def getMissingValueTreatment: String = $(missingValueTreatment)

	/**
	 * @group setParam
	 */
	def setMissingValueTreatment(value: String): T = {
		set(missingValueTreatment, value)
		self
	}

	/**
	 * @group getParam
	 */
	def getMissingValueReplacement: Object = $(missingValueReplacement)

	/**
	 * @group setParam
	 */
	def setMissingValueReplacement(value: Object): T = {
		set(missingValueReplacement, value)
		self
	}

	/**
	 * @group getParam
	 */
	def getInvalidValueTreatment: String = $(invalidValueTreatment)

	/**
	 * @group setParam
	 */
	def setInvalidValueTreatment(value: String): T = {
		set(invalidValueTreatment, value)
		self
	}

	/**
	 * @group getParam
	 */
	def getInvalidValueReplacement: Object = $(invalidValueReplacement)

	/**
	 * @group setParam
	 */
	def setInvalidValueReplacement(value: Object): T = {
		set(invalidValueReplacement, value)
		self
	}

	/**
	 * @group getParam
	 */
	def getWithData: Boolean = $(withData)

	/**
	 * @group setParam
	 */
	def setWithData(value: Boolean): T = {
		set(withData, value)
		self
	}


	protected
	def isMissing(colName: String, col: Column): Column = {
		val missingValuesSet = if(isDefined(missingValues)){
			getMissingValues.toSet
		} else

		{
			Set.empty[Object]
		} // End if

		if(missingValuesSet.nonEmpty){
			col.isin(missingValuesSet.toSeq: _*)
		} else

		{
			col.isNull
		}
	}

	protected
	def isValid(colName: String, col: Column, isNotMissingCol: Column): Column = {
		isNotMissingCol
	}

	protected
	def isInvalid(colName: String, col: Column, isMissingCol: Column, isValidCol: Column): Column = {
		not(isMissingCol) && not(isValidCol)
	}

	protected 
	def validateParams(): Unit = {
		require(isDefined(inputCols) && isDefined(outputCols), "inputCols and outputCols must be defined")
		require(getInputCols.length == getOutputCols.length, "inputCols and outputCols must have the same length")

		if(getOrDefault(missingValueReplacement) != null){

			if(getMissingValueTreatment == MissingValueTreatment.ReturnInvalid.name){
				throw new IllegalArgumentException(s"Missing value treatment ${getMissingValueTreatment} does not support missingValueReplacement")
			}
		} // End if

		if(getOrDefault(invalidValueReplacement) != null){

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

/**
 * @tparam E Self-type
 * @tparam M Model type
 *
 * @param uid Identifier
 */
abstract
class Domain[E <: Domain[E, M], M <: DomainModel[M]](override val uid: String) extends Estimator[M] with HasDomainParams[E] with DefaultParamsWritable {

	setDefault(
		missingValues -> Array.empty[Object],
		missingValueTreatment -> MissingValueTreatment.AsIs.name,
		missingValueReplacement -> null,
		invalidValueTreatment -> InvalidValueTreatment.ReturnInvalid.name,
		invalidValueReplacement -> null,
		withData -> true
	)

	override
	def copy(extra: ParamMap): Domain[E, M] = defaultCopy(extra)

	protected
	def selectNonMissing(inputColNames: Array[String]): Seq[Column] = {
		inputColNames.map {
			inputColName => {
				val inputCol = col(inputColName)

				val isMissingCol = isMissing(inputColName, inputCol)

				val isNotMissingCol = not(isMissingCol)

				when(isNotMissingCol, inputCol).as(inputColName)
			}
		}
	}
}

/**
 * @tparam M Model type
 *
 * @param uid Identifier
 */
abstract
class DomainModel[M <: DomainModel[M]](override val uid: String) extends Model[M] with HasDomainParams[M] with DefaultParamsWritable {

	protected
	def transformMissing(col: Column, isMissingCol: Column): Column = {
		val missingValueTreatment = getMissingValueTreatment

		MissingValueTreatment.forName(missingValueTreatment) match {
			case MissingValueTreatment.AsIs => 
				col
			case MissingValueTreatment.AsMean |
				MissingValueTreatment.AsMode |
				MissingValueTreatment.AsMedian |
				MissingValueTreatment.AsValue =>
				when(isMissingCol, lit(getMissingValueReplacement)).otherwise(col)
			case MissingValueTreatment.ReturnInvalid => 
				when(isMissingCol, raise_error(lit("Missing value"))).otherwise(col)
		}
	}

	protected
	def transformValid(col: Column, isValidCol: Column): Column = {
		col
	}

	protected
	def transformInvalid(col: Column, isInvalidCol: Column): Column = {
		val invalidValueTreatment = getInvalidValueTreatment

		InvalidValueTreatment.forName(invalidValueTreatment) match {
			case InvalidValueTreatment.ReturnInvalid =>
				when(isInvalidCol, raise_error(lit("Invalid value"))).otherwise(col)
			case InvalidValueTreatment.AsIs =>
				col
			case InvalidValueTreatment.AsMissing => {
				val nullifiedCol = when(isInvalidCol, lit(null)).otherwise(col)
				transformMissing(nullifiedCol, isInvalidCol)
			}
			case InvalidValueTreatment.AsValue =>
				when(isInvalidCol, lit(getInvalidValueReplacement)).otherwise(col)
		}
	}

	override
	def transform(dataset: Dataset[_]): DataFrame = {
		val inputColNames = getInputCols
		val outputColNames = getOutputCols

		val outputCols = inputColNames.zip(outputColNames).map {
			case (inputColName, outputColName) => {
				val inputCol = col(inputColName)

				val isMissingCol = isMissing(inputColName, inputCol)
				val afterMissingTransformCol = transformMissing(inputCol, isMissingCol)

				val isNotMissingCol = not(isMissingCol)

				val isValidCol = isValid(inputColName, afterMissingTransformCol, isNotMissingCol)
				val afterValidTransformCol = transformValid(afterMissingTransformCol, isValidCol)

				val isInvalidCol = isInvalid(inputColName, afterValidTransformCol, isMissingCol, isValidCol)
				val afterInvalidTransformCol = transformInvalid(afterValidTransformCol, isInvalidCol)

				val outputCol = afterInvalidTransformCol.as(outputColName)

				// XXX
				//println(s"outputCol for ${outputColName}: ${outputCol.expr.sql}")

				outputCol
			}
		}

		dataset.select(dataset("*") +: outputCols: _*)
	}
}
