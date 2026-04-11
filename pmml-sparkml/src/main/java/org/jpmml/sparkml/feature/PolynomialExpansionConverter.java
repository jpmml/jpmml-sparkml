/*
 * Copyright (c) 2026 Villu Ruusmann
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
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Iterables;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.apache.spark.ml.feature.PolynomialExpansion;
import org.dmg.pmml.DataType;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InteractionFeature;
import org.jpmml.converter.PowerFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class PolynomialExpansionConverter extends FeatureConverter<PolynomialExpansion> {

	public PolynomialExpansionConverter(PolynomialExpansion transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		PolynomialExpansion transformer = getTransformer();

		int degree = transformer.getDegree();

		List<Feature> features = encoder.getFeatures(transformer.getInputCol());

		int outputSize = (int)(CombinatoricsUtils.binomialCoefficient(features.size() + degree, degree) - 1);

		List<String> names = formatNames(outputSize, encoder);

		List<Feature> result = new ArrayList<>();

		expand(features, features.size() - 1, degree, Collections.emptyList(), result, names, encoder);

		return result;
	}

	static
	private void expand(List<Feature> features, int lastIdx, int degree, List<Feature> parentTermFeatures, List<Feature> result, List<String> names, SparkMLEncoder encoder){

		if(degree == 0 || lastIdx < 0){

			if(parentTermFeatures.size() == 1){
				Feature feature = Iterables.getOnlyElement(parentTermFeatures);

				result.add(feature);
			} else

			if(parentTermFeatures.size() >= 2){
				String name = names.get(result.size());

				result.add(new InteractionFeature(encoder, name, DataType.DOUBLE, parentTermFeatures));
			}
		} else

		{
			Feature feature = features.get(lastIdx);

			for(int i = 0; i <= degree; i++){
				List<Feature> termFeatures = new ArrayList<>(parentTermFeatures);

				if(i == 1){
					termFeatures.add(0, feature);
				} else

				if(i >= 2){
					Feature powerFeature;

					if(feature instanceof BinaryFeature){
						BinaryFeature binaryFeature = (BinaryFeature)feature;

						powerFeature = binaryFeature;
					} else

					{
						ContinuousFeature continuousFeature = feature.toContinuousFeature();

						powerFeature = new PowerFeature(encoder, continuousFeature, i);
					}

					termFeatures.add(0, powerFeature);
				}

				expand(features, lastIdx - 1, degree - i, termFeatures, result, names, encoder);
			}
		}
	}
}