/*
 * Copyright (c) 2016 Villu Ruusmann
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
import java.util.Map;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;

@Deprecated
public class ConverterUtil {

	private ConverterUtil(){
	}

	static
	public PMML toPMML(StructType schema, PipelineModel pipelineModel){
		return toPMML(schema, pipelineModel, Collections.emptyMap());
	}

	static
	public PMML toPMML(StructType schema, PipelineModel pipelineModel, Map<String, ? extends Map<String, ?>> options){
		throw new UnsupportedOperationException(formatMessage("toPMML", "build"));
	}

	static
	public byte[] toPMMLByteArray(StructType schema, PipelineModel pipelineModel){
		return toPMMLByteArray(schema, pipelineModel, Collections.emptyMap());
	}

	static
	public byte[] toPMMLByteArray(StructType schema, PipelineModel pipelineModel, Map<String, ? extends Map<String, ?>> options){
		throw new UnsupportedOperationException(formatMessage("toPMMLByteArray", "buildByteArray"));
	}

	static
	private String formatMessage(String toPmmlMethod, String buildMethod){
		return "Replace \"" + ConverterUtil.class.getName() + "." + toPmmlMethod + "(schema, pipelineModel)\" with \"new " + PMMLBuilder.class.getName() + "(schema, pipelineModel)." + buildMethod + "()\"";
	}
}
