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

import java.lang.{
	Boolean => JavaBoolean,
	Double => JavaDouble,
	Float => JavaFloat, 
	Integer => JavaInteger,
	Long => JavaLong,
}

import org.apache.spark.ml.param.{Param, Params}
import org.json4s._
import org.json4s.jackson.JsonMethods.{compact, parse, render}

private[feature] object ValueMapper {

	private
	val numberClasses: Set[Class[_]] = Set(
		classOf[JavaInteger],
		classOf[JavaLong],
		classOf[JavaFloat],
		classOf[JavaDouble]
	)

	private
	val objectClasses: Set[Class[_]] = Set(
		classOf[JavaInteger],
		classOf[JavaLong],
		classOf[JavaFloat],
		classOf[JavaDouble],
		classOf[JavaBoolean],
		classOf[String]
	)

	def isNumber(value: AnyRef): Boolean = {
		value == null || numberClasses.contains(value.getClass)
	}

	def isObject(value: AnyRef): Boolean = {
		value == null || objectClasses.contains(value.getClass)
	}

	def javaToJS(value: Any): JValue = value match {
		case null => 
			JNull
		case integer: JavaInteger => 
			JLong(integer.intValue())
		case _long: JavaLong => 
			JLong(_long.longValue())
		case _float: JavaFloat => 
			JDouble(_float.floatValue())
		case _double: JavaDouble => 
			JDouble(_double.doubleValue())
		case _boolean: JavaBoolean =>
			JBool(_boolean.booleanValue())
		case string: String => 
			JString(string)
		case _ =>
			throw new IllegalArgumentException()
	}

	def jsToJava(jsValue: JValue): AnyRef = jsValue match {
		case JNull =>
			null
		case JLong(value) =>
			JavaLong.valueOf(value)
		case JDouble(value) =>
			JavaDouble.valueOf(value)
		case JBool(value) =>
			JavaBoolean.valueOf(value)
		case JString(value) => 
			value
		case _ =>
			throw new IllegalArgumentException()
	}
}

class ScalarParam[T <: AnyRef](parent: Params, name: String, doc: String, isValid: T => Boolean) extends Param[T](parent, name, doc, isValid){

	def this(parent: Params, name: String, doc: String) = this(parent, name, doc, _ => true)

	override 
	def jsonEncode(value: T): String = {
		val jsValue = ValueMapper.javaToJS(value)
		compact(render(jsValue))
	}

	override 
	def jsonDecode(json: String): T = {
		val jsValue = parse(json)
		ValueMapper.jsToJava(jsValue).asInstanceOf[T]
	}
}

object ScalarParam {

	def numberParam(parent: Params, name: String, doc: String): ScalarParam[Number] = {
		new ScalarParam[Number](parent, name, doc, ValueMapper.isNumber)
	}

	def objectParam(parent: Params, name: String, doc: String): ScalarParam[AnyRef] = {
		new ScalarParam[AnyRef](parent, name, doc, ValueMapper.isObject)
	}
}

class ScalarArrayParam[T <: AnyRef](parent: Params, name: String, doc: String, isValid: Array[T] => Boolean) extends Param[Array[T]](parent, name, doc, isValid){

	def this(parent: Params, name: String, doc: String) = this(parent, name, doc, _ => true)

	override
	def jsonEncode(value: Array[T]): String = {
		val jsValues = value.map(ValueMapper.javaToJS)
		compact(render(JArray(jsValues.toList)))
	}

	override
	def jsonDecode(json: String): Array[T] = {
		val jsValue = parse(json)
		jsValue match {
			case JArray(jsValues) =>
				jsValues.map(ValueMapper.jsToJava).toArray.asInstanceOf[Array[T]]
			case _ =>
				throw new IllegalArgumentException()
		}
	}
}

object ScalarArrayParam {

	def objectArrayParam(parent: Params, name: String, doc: String): ScalarArrayParam[AnyRef] = {
		new ScalarArrayParam[AnyRef](parent, name, doc, value => value.forall(ValueMapper.isObject))
	}
}

class MapParam[T <: AnyRef](parent: Params, name: String, doc: String, isValid: Map[String, Array[T]] => Boolean) extends Param[Map[String, Array[T]]](parent, name, doc, isValid){

	def this(parent: Params, name: String, doc: String) = this(parent, name, doc, _ => true)

	override
	def jsonEncode(value: Map[String, Array[T]]): String = {
		val fields = value.map {
			case (k, v) => {
				k -> JArray(v.map(ValueMapper.javaToJS).toList)
			}
		}
		val jsValue = JObject(fields.toList)
		compact(render(jsValue))
	}

	override
	def jsonDecode(json: String): Map[String, Array[T]] = {
		val jsValue = parse(json)
		jsValue match {
			case JObject(fields) => {
				fields.map {
					case (k, JArray(jsValues)) => {
						k -> jsValues.map(ValueMapper.jsToJava).toArray.asInstanceOf[Array[T]]
					}
					case (k, _) =>
						throw new IllegalArgumentException()
				}.toMap
			}
			case _ =>
				throw new IllegalArgumentException()
		}
	}
}

object MapParam {

	def numberArrayMapParam(parent: Params, name: String, doc: String): MapParam[Number] = {
		new MapParam[Number](parent, name, doc, value => value.forall {
			case (_, array) => array.forall(ValueMapper.isNumber)
		})
	}

	def objectArrayMapParam(parent: Params, name: String, doc: String): MapParam[Object] = {
		new MapParam[Object](parent, name, doc, value => value.forall {
			case (_, array) => array.forall(ValueMapper.isObject)
		})
	}
}