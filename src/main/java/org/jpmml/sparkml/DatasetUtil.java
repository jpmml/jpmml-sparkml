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

import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalog.Catalog;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.execution.QueryExecution;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.IntegralType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataType;

public class DatasetUtil {

	private DatasetUtil(){
	}

	static
	public LogicalPlan createAnalyzedLogicalPlan(SparkSession sparkSession, StructType schema, String statement){
		String tableName = "sql2pmml_" + DatasetUtil.ID.getAndIncrement();

		statement = statement.replace("__THIS__", tableName);

		Dataset<Row> dataset = sparkSession.createDataFrame(Collections.emptyList(), schema);

		dataset.createOrReplaceTempView(tableName);

		try {
			QueryExecution queryExecution = sparkSession.sql(statement).queryExecution();

			return queryExecution.analyzed();
		} finally {
			Catalog catalog = sparkSession.catalog();

			catalog.dropTempView(tableName);
		}
	}

	static
	public DataType translateDataType(org.apache.spark.sql.types.DataType sparkDataType){

		if(sparkDataType instanceof StringType){
			return DataType.STRING;
		} else

		if(sparkDataType instanceof IntegralType){
			return DataType.INTEGER;
		} else

		if(sparkDataType instanceof DoubleType){
			return DataType.DOUBLE;
		} else

		if(sparkDataType instanceof BooleanType){
			return DataType.BOOLEAN;
		} else

		{
			throw new IllegalArgumentException("Expected string, integral, double or boolean data type, got " + sparkDataType.typeName() + " data type");
		}
	}

	private static final AtomicInteger ID = new AtomicInteger(1);
}