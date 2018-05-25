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

import java.util.List;
import java.util.ListIterator;

import org.jpmml.converter.Feature;

public class ScaledFeatureUtil {

	private ScaledFeatureUtil(){
	}

	static
	public void simplify(List<Feature> features, List<Double> coefficients){

		if(features.size() != coefficients.size()){
			throw new IllegalArgumentException();
		}

		ListIterator<Feature> featureIt = features.listIterator();
		ListIterator<Double> coefficientIt = coefficients.listIterator();

		while(featureIt.hasNext()){
			Feature feature = featureIt.next();
			Double coefficient = coefficientIt.next();

			if(feature instanceof ScaledFeature){
				ScaledFeature scaledFeature = (ScaledFeature)feature;

				featureIt.set(scaledFeature.getFeature());
				coefficientIt.set(coefficient * scaledFeature.getFactor());
			}
		}
	}
}