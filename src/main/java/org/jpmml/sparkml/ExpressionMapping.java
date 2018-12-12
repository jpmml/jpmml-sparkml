/*
 * Copyright (c) 2018 Villu Ruusmann
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

import org.apache.spark.sql.catalyst.expressions.Expression;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;

public class ExpressionMapping {

	private Expression from = null;

	private org.dmg.pmml.Expression to = null;

	private DataType dataType = null;


	public ExpressionMapping(Expression from, org.dmg.pmml.Expression to, DataType dataType){
		setFrom(from);
		setTo(to);
		setDataType(dataType);
	}

	public Expression getFrom(){
		return this.from;
	}

	private void setFrom(Expression from){
		this.from = from;
	}

	public org.dmg.pmml.Expression getTo(){
		return this.to;
	}

	private void setTo(org.dmg.pmml.Expression to){
		this.to = to;
	}

	public DataType getDataType(){
		return this.dataType;
	}

	private void setDataType(DataType dataType){
		this.dataType = dataType;
	}

	public OpType getOpType(){
		DataType dataType = getDataType();

		switch(dataType){
			case STRING:
				return OpType.CATEGORICAL;
			case INTEGER:
			case DOUBLE:
				return OpType.CONTINUOUS;
			case BOOLEAN:
				return OpType.CATEGORICAL;
			default:
				throw new IllegalArgumentException();
		}
	}
}