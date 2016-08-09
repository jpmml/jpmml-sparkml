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
import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.feature.Bucketizer;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Discretize;
import org.dmg.pmml.DiscretizeBin;
import org.dmg.pmml.Interval;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ListFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;

public class BucketizerConverter extends FeatureConverter<Bucketizer> {

	public BucketizerConverter(Bucketizer transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		Bucketizer transformer = getTransformer();

		ContinuousFeature inputFeature = (ContinuousFeature)featureMapper.getOnlyFeature(transformer.getInputCol());

		Discretize discretize = new Discretize(inputFeature.getName());

		List<String> categories = new ArrayList<>();

		double[] splits = transformer.getSplits();
		for(int i = 0; i < (splits.length - 1); i++){
			String category = String.valueOf(i);

			categories.add(category);

			Interval interval = new Interval((i < (splits.length - 2)) ? Interval.Closure.CLOSED_OPEN : Interval.Closure.CLOSED_CLOSED)
				.setLeftMargin(formatMargin(splits[i]))
				.setRightMargin(formatMargin(splits[i + 1]));

			DiscretizeBin discretizeBin = new DiscretizeBin(category, interval);

			discretize.addDiscretizeBins(discretizeBin);
		}

		DerivedField derivedField = featureMapper.createDerivedField(formatName(transformer), OpType.CONTINUOUS, DataType.INTEGER, discretize);

		Feature feature = new ListFeature(derivedField, categories);

		return Collections.singletonList(feature);
	}

	static
	private Double formatMargin(double value){

		if(Double.isInfinite(value)){
			return null;
		}

		return value;
	}
}