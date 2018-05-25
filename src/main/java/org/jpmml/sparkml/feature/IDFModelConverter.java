/*
 * Copyright (c) 2017 Villu Ruusmann
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
import java.util.List;

import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.linalg.Vector;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.ScaledFeature;
import org.jpmml.sparkml.SparkMLEncoder;
import org.jpmml.sparkml.TermFeature;
import org.jpmml.sparkml.WeightedTermFeature;

public class IDFModelConverter extends FeatureConverter<IDFModel> {

	public IDFModelConverter(IDFModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		IDFModel transformer = getTransformer();

		List<Feature> features = encoder.getFeatures(transformer.getInputCol());

		Vector idf = transformer.idf();
		if(idf.size() != features.size()){
			throw new IllegalArgumentException();
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);
			Double weight = idf.apply(i);

			ScaledFeature scaledFeature = new ScaledFeature(encoder, feature, weight){

				private WeightedTermFeature weightedTermFeature = null;


				@Override
				public ContinuousFeature toContinuousFeature(){

					if(this.weightedTermFeature == null){
						TermFeature termFeature = (TermFeature)getFeature();
						Double factor = getFactor();

						this.weightedTermFeature = termFeature.toWeightedTermFeature(factor);
					}

					return this.weightedTermFeature.toContinuousFeature();
				}
			};

			result.add(scaledFeature);
		}

		return result;
	}
}