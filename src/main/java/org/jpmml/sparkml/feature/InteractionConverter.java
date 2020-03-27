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
package org.jpmml.sparkml.feature;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.Interaction;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InteractionFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class InteractionConverter extends FeatureConverter<Interaction> {

	public InteractionConverter(Interaction transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		Interaction transformer = getTransformer();

		StringBuilder sb = new StringBuilder();

		List<Feature> result = new ArrayList<>();

		String[] inputCols = transformer.getInputCols();
		for(int i = 0; i < inputCols.length; i++){
			String inputCol = inputCols[i];

			List<Feature> features = encoder.getFeatures(inputCol);

			if(features.size() == 1){
				Feature feature = features.get(0);

				categorical:
				if(feature instanceof CategoricalFeature){
					CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

					FieldName name = categoricalFeature.getName();

					DataType dataType = categoricalFeature.getDataType();
					switch(dataType){
						case INTEGER:
							break;
						case FLOAT:
						case DOUBLE:
							break categorical;
						default:
							break;
					}

					// XXX
					inputCol = name.getValue();

					features = (List)OneHotEncoderModelConverter.encodeFeature(categoricalFeature.getEncoder(), categoricalFeature, categoricalFeature.getValues(), false);
				}
			} // End if

			if(i == 0){
				sb.append(inputCol);

				result = features;
			} else

			{
				sb.append(':').append(inputCol);

				List<Feature> interactionFeatures = new ArrayList<>();

				int index = 0;

				for(Feature left : result){

					for(Feature right : features){
						interactionFeatures.add(new InteractionFeature(encoder, FieldName.create(sb.toString() + "[" + index + "]"), DataType.DOUBLE, Arrays.asList(left, right)));

						index++;
					}
				}

				result = interactionFeatures;
			}
		}

		return result;
	}
}