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
import org.dmg.pmml.FieldName;
import org.jpmml.converter.Feature;

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
			if(features.size() == 1){
				Feature feature = features.get(0);

				if(feature instanceof BinarizedCategoricalFeature){
					BinarizedCategoricalFeature binarizedCategoricalFeature = (BinarizedCategoricalFeature)feature;

					features = (List)binarizedCategoricalFeature.getBinaryFeatures();
				}
			}

			encoder.putFeatures(outputCol, features);
		} else

		if((InOutMode.MULTIPLE).equals(outputMode)){
			HasOutputCols hasOutputCols = (HasOutputCols)transformer;

			String[] outputCols = hasOutputCols.getOutputCols();

			List<Feature> features = encodeFeatures(encoder);
			if(outputCols.length != features.size()){
				throw new IllegalArgumentException("Expected " + outputCols.length + " features, got " + features.size() + " features");
			}

			for(int i = 0; i < outputCols.length; i++){
				String outputCol = outputCols[i];
				Feature feature = features.get(i);

				if(feature instanceof BinarizedCategoricalFeature){
					BinarizedCategoricalFeature binarizedCategoricalFeature = (BinarizedCategoricalFeature)feature;

					encoder.putFeatures(outputCol, (List)binarizedCategoricalFeature.getBinaryFeatures());
				} else

				{
					encoder.putOnlyFeature(outputCol, feature);
				}
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
	protected enum InOutMode {
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
	public <T extends Transformer & HasOutputCol> FieldName formatName(T transformer){
		return FieldName.create(transformer.getOutputCol());
	}

	static
	public <T extends Transformer & HasOutputCol & HasOutputCols> FieldName formatName(T transformer, int index){

		if(transformer.isSet(transformer.outputCols())){
			return FieldName.create(transformer.getOutputCols()[index]);
		} // End if

		if(index != 0){
			throw new IllegalArgumentException();
		}

		return FieldName.create(transformer.getOutputCol());
	}

	static
	public <T extends Transformer & HasOutputCol> FieldName formatName(T transformer, int index, int length){

		if(length > 1){
			return FieldName.create(transformer.getOutputCol() + "[" + index + "]");
		}

		return FieldName.create(transformer.getOutputCol());
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