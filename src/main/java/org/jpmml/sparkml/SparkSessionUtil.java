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

import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;

public class SparkSessionUtil {

	private SparkSessionUtil(){
	}

	static
	public SparkSession createSparkSession(){
		SparkSession.Builder builder = SparkSession.builder()
			.appName("test")
			.master("local[1]")
			.config("spark.ui.enabled", false);

		SparkSession sparkSession = builder.getOrCreate();

		SparkContext sparkContext = sparkSession.sparkContext();
		sparkContext.setLogLevel("ERROR");

		return sparkSession;
	}

	static
	public SparkSession destroySparkSession(SparkSession sparkSession){
		sparkSession.stop();

		return null;
	}
}