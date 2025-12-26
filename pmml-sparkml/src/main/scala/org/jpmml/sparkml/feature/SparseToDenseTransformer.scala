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

import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

@deprecated("Use VectorDensifier instead", "3.2.7")
class SparseToDenseTransformer(override val uid: String) extends VectorDensifier(uid) {

	def this() = this(Identifiable.randomUID("sparse2dense"))
}

object SparseToDenseTransformer extends DefaultParamsReadable[SparseToDenseTransformer] {

	def sparseToDense(vec: Vector): DenseVector = {
		VectorDensifier.toDense(vec)
	}
}