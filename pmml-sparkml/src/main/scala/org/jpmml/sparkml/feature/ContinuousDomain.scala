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

import org.apache.spark.ml.param.{Param, ParamMap, Params, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{Column, Dataset}
import org.apache.spark.sql.functions.{col, isnan, lit, max, min, not, when}

sealed
trait OutlierTreatment {
	def name: String
}

object OutlierTreatment {
	case object AsIs extends OutlierTreatment { 
		val name = "asIs"
	}

	case object AsMissingValues extends OutlierTreatment {
		val name = "asMissingValues"
	}

	case object AsExtremeValues extends OutlierTreatment {
		val name = "asExtremeValues" 
	}

	val values: Array[OutlierTreatment] = Array(AsIs, AsMissingValues, AsExtremeValues)

	def forName(name: String): OutlierTreatment = values
		.find(_.name == name)
		.getOrElse(throw new IllegalArgumentException(name))
}

trait HasContinuousDomainParams[T <: HasContinuousDomainParams[T]] extends HasDomainParams[T] {

	/**
	 * @group param
	 */
	val outlierTreatment: Param[String] = new Param[String](this, "outlierTreatment", "", ParamValidators.inArray(OutlierTreatment.values.map(_.name)))

	/**
	 * @group param
	 */
	val lowValue: Param[Number] = new Param[Number](this, "lowValue", "")

	/**
	 * @group param
	 */
	val highValue: Param[Number] = new Param[Number](this, "highValue", "")

	/**
	 * @group param
	 */
	val dataRanges: Param[Map[String, Array[Number]]] = new Param[Map[String, Array[Number]]](this, "dataRanges", "")


	/**
	 * @group getParam
	 */
	def getOutlierTreatment(): String = $(outlierTreatment)

	/**
	 * @group setParam
	 */
	def setOutlierTreatment(value: String): T = {
		set(outlierTreatment, value)
		self
	}

	/**
	 * @group getParam
	 */
	def getLowValue(): Number = $(lowValue)

	/**
	 * @group setParam
	 */
	def setLowValue(value: Number): T = {
		set(lowValue, value)
		self
	}

	/**
	 * @group getParam
	 */
	def getHighValue(): Number = $(highValue)

	/**
	 * @group setParam
	 */
	def setHighValue(value: Number): T = {
		set(highValue, value)
		self
	}

	/**
	 * @group getParam
	 */
	def getDataRanges(): Map[String, Array[Number]] = $(dataRanges)

	/**
	 * @group setParam
	 */
	def setDataRanges(value: Map[String, Array[Number]]): T = {
		set(dataRanges, value)
		self
	}


	override
	def validateParams(): Unit = {
		super.validateParams()

		if(getOutlierTreatment == OutlierTreatment.AsIs.name){
			
			if(isDefined(lowValue) || isDefined(highValue)){
				throw new IllegalArgumentException(s"Outlier treatment ${getOutlierTreatment} does not support lowValue or highValue")
			}
		} else

		if(getOutlierTreatment == OutlierTreatment.AsMissingValues.name || getOutlierTreatment == OutlierTreatment.AsExtremeValues.name){
		
			if(!isDefined(lowValue) || !isDefined(highValue)){
				throw new IllegalArgumentException(s"Outlier treatment ${getOutlierTreatment} requires lowValue and highValue")
			}
		} // End if

		if(getDataRanges.nonEmpty){
			require(getDataRanges.keys.forall(getInputCols.toSet.contains), s"dataRanges keys and inputCols must match")

			if(!getWithData){
				throw new IllegalArgumentException("dataRanges requires withData")
			}
		}
	}
}

class ContinuousDomain(override val uid: String) extends Domain[ContinuousDomain, ContinuousDomainModel](uid) with HasContinuousDomainParams[ContinuousDomain] {

	override
	def setInputCols(value: Array[String]): ContinuousDomain = super.setInputCols(value)

	override
	def setOutputCols(value: Array[String]): ContinuousDomain = super.setOutputCols(value)


	def this() = this(Identifiable.randomUID("contDomain"))

	setDefault(
		outlierTreatment -> OutlierTreatment.AsIs.name,
		dataRanges -> Map.empty[String, Array[Number]]
	)

	override
	def fit(dataset: Dataset[_]): ContinuousDomainModel = {
		val fitDataRanges: Map[String, Array[Number]] = if(getWithData){
			if(getDataRanges.nonEmpty){
				getDataRanges
			} else

			{
				collectDataRanges(dataset)
			}
		} else

		{
			Map.empty[String, Array[Number]]
		}

		val model = new ContinuousDomainModel(uid)
			.setDataRanges(fitDataRanges)

		copyValues(model)
	}

	protected
	def collectDataRanges(dataset: Dataset[_]): Map[String, Array[Number]] = {
		val inputColNames = getInputCols

		val selectCols = selectNonMissing(inputColNames)

		val aggCols = inputColNames.flatMap {
			inputColName => {
				val inputCol = col(inputColName)

				Seq(
					min(when(!isnan(inputCol), inputCol)),
					max(when(!isnan(inputCol), inputCol))
				)
			}
		}

		val dataRanges = dataset
			.select(selectCols: _*)
			.groupBy()
			.agg(aggCols.head, aggCols.tail: _*)
			.collect()
			.headOption
			.map {
				row => inputColNames.zipWithIndex.map {
					case (colName, idx) => {
						val minIdx = 2 * idx
						val maxIdx = 2 * idx + 1
						colName -> Array(row.getAs[Number](minIdx), row.getAs[Number](maxIdx))
					}
				}.toMap
			}
			.getOrElse(Map.empty[String, Array[Number]])

		dataRanges
	}
}

object ContinuousDomain extends DefaultParamsReadable[ContinuousDomain]

class ContinuousDomainModel(override val uid: String) extends DomainModel[ContinuousDomainModel](uid) with HasContinuousDomainParams[ContinuousDomainModel] {

	override
	protected
	def isValid(colName: String, col: Column, isNotMissingCol: Column): Column = {
		
		if(getDataRanges.nonEmpty){
			val (dataMin, dataMax) = getDataRanges.get(colName) match {
				case Some(Array(min, max)) =>
					(lit(min), lit(max))
				case _ =>
					// XXX
					return lit(false)
			}

			(col >= dataMin) && (col <= dataMax) && !isnan(col)
		} else

		{
			super.isValid(colName, col, isNotMissingCol) && !isnan(col)
		}
	}

	override
	protected
	def transformValid(col: Column, isValidCol: Column): Column = {
		val outlierTreatment = getOutlierTreatment

		OutlierTreatment.forName(outlierTreatment) match {
			case OutlierTreatment.AsIs =>
				col
			case OutlierTreatment.AsMissingValues => {
				val isOutlier = (col < getLowValue) || (col > getHighValue)
				val nullifiedCol = when(isOutlier, lit(null)).otherwise(col)
				transformMissing(nullifiedCol, isOutlier)
			}
			case OutlierTreatment.AsExtremeValues => {
				val isNegativeOutlier = (col < getLowValue)
				val isPositiveOutlier = (col > getHighValue)
				when(isNegativeOutlier, lit(getLowValue)).otherwise(when(isPositiveOutlier, lit(getHighValue)).otherwise(col))
			}
			case _ =>
				throw new IllegalArgumentException(outlierTreatment)
		}
	}

	override
	def copy(extra: ParamMap): ContinuousDomainModel = {
		defaultCopy(extra)
	}
}

object ContinuousDomainModel extends DefaultParamsReadable[ContinuousDomainModel]
