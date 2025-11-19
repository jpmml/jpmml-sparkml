/*
 * Copyright (c) 2020 Villu Ruusmann
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

import java.util.List;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.shared.HasInputCol;
import org.apache.spark.ml.param.shared.HasInputCols;
import org.apache.spark.ml.param.shared.HasOutputCol;
import org.apache.spark.ml.param.shared.HasOutputCols;

abstract
public class MultiFeatureConverter<T extends Transformer & HasInputCol & HasInputCols & HasOutputCol & HasOutputCols> extends FeatureConverter<T> {

	public MultiFeatureConverter(T transformer){
		super(transformer);
	}

	protected String formatMultiName(int index, int length, SparkMLEncoder encoder){
		T transformer = getTransformer();

		if(transformer.isSet(transformer.outputCols())){
			String[] outputCols = transformer.getOutputCols();

			String outputCol = outputCols[index];

			return encoder.mapOnlyFieldName(outputCol);
		}

		if(index != 0){
			throw new IllegalArgumentException();
		}

		List<String> names = formatNames(length, encoder);

		return names.get(index);
	}

	@Override
	protected InOutMode getInputMode(){
		T transformer = getTransformer();

		return getInputMode(transformer);
	}

	@Override
	public InOutMode getOutputMode(){
		return getInputMode();
	}
}