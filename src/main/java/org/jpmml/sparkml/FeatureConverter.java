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

		OutputMode outputMode = getOutputMode();

		if((OutputMode.SINGLE).equals(outputMode)){
			HasOutputCol hasOutputCol = (HasOutputCol)transformer;

			String outputCol = hasOutputCol.getOutputCol();

			List<Feature> features = encodeFeatures(encoder);

			encoder.putFeatures(outputCol, features);
		} else

		if((OutputMode.MULTIPLE).equals(outputMode)){
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

	protected OutputMode getOutputMode(){
		T transformer = getTransformer();

		if(transformer instanceof HasOutputCol){
			return OutputMode.SINGLE;
		} else

		if(transformer instanceof HasOutputCols){
			return OutputMode.MULTIPLE;
		}

		return null;
	}

	static
	protected enum OutputMode {
		SINGLE,
		MULTIPLE,
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
}