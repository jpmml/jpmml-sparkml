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

import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{Column, Dataset}
import org.apache.spark.sql.functions.collect_set

trait HasCategoricalDomainParams[T <: HasCategoricalDomainParams[T]] extends HasDomainParams[T] {

	/**
	 * @group param
	 */
	val dataValues: Param[Map[String, Array[Object]]] = MapParam.objectArrayMapParam(this, "dataValues", "")


	/**
	 * @group getParam
	 */
	def getDataValues: Map[String, Array[Object]] = $(dataValues)

	/**
	 * @group setParam
	 */
	def setDataValues(value: Map[String, Array[Object]]): T = {
		set(dataValues, value)
		self
	}


	override
	def validateParams(): Unit = {
		super.validateParams()

		if(getDataValues.nonEmpty){
			require(getDataValues.keys.forall(getInputCols.toSet.contains), s"dataValues keys and inputCols must match")

			if(!getWithData){
				throw new IllegalArgumentException("dataValues requires withData")
			}
		}
	}
}

class CategoricalDomain(override val uid: String) extends Domain[CategoricalDomain, CategoricalDomainModel](uid) with HasCategoricalDomainParams[CategoricalDomain] {

	override
	def setInputCols(value: Array[String]): CategoricalDomain = super.setInputCols(value)

	override
	def setOutputCols(value: Array[String]): CategoricalDomain = super.setOutputCols(value)


	def this() = this(Identifiable.randomUID("catDomain"))

	setDefault(
		dataValues -> Map.empty[String, Array[Object]]
	)

	override
	def fit(dataset: Dataset[_]): CategoricalDomainModel = {
		val fitDataValues: Map[String, Array[Object]] = if(getWithData){
			if(getDataValues.nonEmpty){
				getDataValues
			} else

			{
				collectDataValues(dataset)
			}
		} else

		{
			Map.empty[String, Array[Object]]
		}

		val model = new CategoricalDomainModel(uid)
			.setDataValues(fitDataValues)

		copyValues(model)
	}

	protected def collectDataValues(dataset: Dataset[_]): Map[String, Array[Object]] = {
		val inputColNames = getInputCols

		val selectCols = selectNonMissing(inputColNames)

		val aggCols = inputColNames.map {
			inputColName => collect_set(inputColName).as(inputColName)
		}

		val dataValues = dataset
			.select(selectCols: _*)
			.groupBy()
			.agg(aggCols.head, aggCols.tail: _*)
			.collect()
			.headOption
			.map {
				row => inputColNames.zipWithIndex.map {
					case (colName, idx) =>
						val seq = row.getAs[scala.collection.Seq[Object]](idx)
						colName -> seq.toArray[Object]
				}.toMap
			}
			.getOrElse(Map.empty[String, Array[Object]])

		dataValues
	}
}

object CategoricalDomain extends DefaultParamsReadable[CategoricalDomain]

class CategoricalDomainModel(override val uid: String) extends DomainModel[CategoricalDomainModel](uid) with HasCategoricalDomainParams[CategoricalDomainModel] {

	override
	protected def isValid(colName: String, col: Column, isNotMissingCol: Column): Column = {

		if(getDataValues.nonEmpty){
			val values = getDataValues.getOrElse(colName, Array.empty[Object])

			if(values.nonEmpty){
				val valuesSet = values.toSet

				isNotMissingCol && col.isin(valuesSet.toSeq: _*)
			} else

			{
				super.isValid(colName, col, isNotMissingCol)
			}
		} else

		{
			super.isValid(colName, col, isNotMissingCol)
		}
	}

	override 
	def copy(extra: ParamMap): CategoricalDomainModel = {
		defaultCopy(extra)
	}
}

object CategoricalDomainModel extends DefaultParamsReadable[CategoricalDomainModel]
