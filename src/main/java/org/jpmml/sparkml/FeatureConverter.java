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

import java.util.List;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.shared.HasInputCol;
import org.apache.spark.ml.param.shared.HasInputCols;
import org.apache.spark.ml.param.shared.HasOutputCol;
import org.apache.spark.ml.param.shared.HasOutputCols;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;

abstract
public class FeatureConverter<T extends Transformer> extends TransformerConverter<T> {

	public FeatureConverter(T transformer){
		super(transformer);
	}

	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		throw new UnsupportedOperationException();
	}

	public void registerFeatures(SparkMLEncoder encoder){
		Transformer transformer = getTransformer();

		InOutMode outputMode = getOutputMode();

		if((InOutMode.SINGLE).equals(outputMode)){
			HasOutputCol hasOutputCol = (HasOutputCol)transformer;

			String outputCol = hasOutputCol.getOutputCol();

			List<Feature> features = encodeFeatures(encoder);

			encoder.putFeatures(outputCol, features);
		} else

		if((InOutMode.MULTIPLE).equals(outputMode)){
			HasOutputCols hasOutputCols = (HasOutputCols)transformer;

			String[] outputCols = hasOutputCols.getOutputCols();

			List<Feature> features = encodeFeatures(encoder);

			SchemaUtil.checkSize(outputCols.length, features);

			for(int i = 0; i < outputCols.length; i++){
				String outputCol = outputCols[i];
				Feature feature = features.get(i);

				encoder.putOnlyFeature(outputCol, feature);
			}
		}
	}

	protected InOutMode getInputMode(){
		throw new IllegalArgumentException();
	}

	protected InOutMode getOutputMode(){
		T transformer = getTransformer();

		return getOutputMode(transformer);
	}

	static
	public enum InOutMode {
		SINGLE(){

			@Override
			public <T extends Transformer & HasInputCol & HasInputCols> String[] getInputCols(T transformer){
				return new String[]{transformer.getInputCol()};
			}

			@Override
			public <T extends Transformer> String[] getOutputCols(T transformer){

				if(transformer instanceof HasOutputCol){
					HasOutputCol hasOutputCol = (HasOutputCol)transformer;

					return new String[]{hasOutputCol.getOutputCol()};
				}

				throw new IllegalArgumentException();
			}
		},

		MULTIPLE(){

			@Override
			public <T extends Transformer & HasInputCol & HasInputCols> String[] getInputCols(T transformer){
				return transformer.getInputCols();
			}

			@Override
			public <T extends Transformer> String[] getOutputCols(T transformer){

				if(transformer instanceof HasOutputCols){
					HasOutputCols hasOutputCols = (HasOutputCols)transformer;

					return hasOutputCols.getOutputCols();
				}

				throw new IllegalArgumentException();
			}
		},
		;

		abstract
		public <T extends Transformer & HasInputCol & HasInputCols> String[] getInputCols(T transformer);

		abstract
		public <T extends Transformer> String[] getOutputCols(T transformer);
	}

	static
	public <T extends Transformer & HasOutputCol> String formatName(T transformer){
		return transformer.getOutputCol();
	}

	static
	public <T extends Transformer & HasOutputCol> String formatName(T transformer, int index, int length){

		if(length > 1){
			return transformer.getOutputCol() + ("[" + index + "]");
		}

		return transformer.getOutputCol();
	}

	static
	protected <T extends HasInputCol & HasInputCols> InOutMode getInputMode(T transformer){

		if(transformer.isSet(transformer.inputCol())){
			return InOutMode.SINGLE;
		} else

		if(transformer.isSet(transformer.inputCols())){
			return InOutMode.MULTIPLE;
		}

		throw new IllegalArgumentException();
	}

	static
	protected <T extends Transformer> InOutMode getOutputMode(T transformer){

		if(transformer instanceof HasOutputCol){
			return InOutMode.SINGLE;
		} else

		if(transformer instanceof HasOutputCols){
			return InOutMode.MULTIPLE;
		}

		return null;
	}
}