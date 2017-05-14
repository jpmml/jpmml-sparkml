/*
 * Copyright (c) 2017 Villu Ruusmann
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
package org.jpmml.sparkml;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.linalg.Matrix;

public class MatrixUtil {

	private MatrixUtil(){
	}

	static
	public List<Double> getRow(Matrix matrix, int row){
		List<Double> result = new ArrayList<>();

		for(int column = 0; column < matrix.numCols(); column++){
			result.add(matrix.apply(row, column));
		}

		return result;
	}

	static
	public List<Double> getColumn(Matrix matrix, int column){
		List<Double> result = new ArrayList<>();

		for(int row = 0; row < matrix.numRows(); row++){
			result.add(matrix.apply(row, column));
		}

		return result;
	}
}