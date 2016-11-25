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
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;
import org.jpmml.sparkml.FeatureUtil;
import org.jpmml.sparkml.InteractionFeature;

public class InteractionConverter extends FeatureConverter<Interaction> {

	public InteractionConverter(Interaction transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		Interaction transformer = getTransformer();

		String name = "";

		List<Feature> features = new ArrayList<>();

		String[] inputCols = transformer.getInputCols();
		for(int i = 0; i < inputCols.length; i++){
			String inputCol = inputCols[i];

			List<Feature> inputFeatures = featureMapper.getFeatures(inputCol);

			if(i == 0){
				name = inputCol;

				features = inputFeatures;
			} else

			{
				name += (":" + inputCol);

				List<InteractionFeature> interactionFeatures = new ArrayList<>();

				int index = 0;

				for(Feature feature : features){

					for(Feature inputFeature : inputFeatures){

						Apply apply = new Apply("*")
							.addExpressions((FeatureUtil.toContinuousFeature(feature)).ref(), (FeatureUtil.toContinuousFeature(inputFeature)).ref());

						DerivedField derivedField = featureMapper.createDerivedField(FieldName.create(name + "[" + index + "]"), OpType.CONTINUOUS, DataType.DOUBLE, apply);

						InteractionFeature interactionFeature = new InteractionFeature(derivedField, Arrays.asList(feature, inputFeature));

						interactionFeatures.add(interactionFeature);

						index++;
					}
				}

				features = (List)interactionFeatures;
			}
		}

		return features;
	}
}