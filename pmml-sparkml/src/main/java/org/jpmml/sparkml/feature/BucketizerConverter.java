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
import org.jpmml.converter.IndexFeature;
import org.jpmml.sparkml.MultiFeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class BucketizerConverter extends MultiFeatureConverter<Bucketizer> {

	public BucketizerConverter(Bucketizer transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		Bucketizer transformer = getTransformer();

		InOutMode inputMode = getInputMode();

		String[] inputCols;
		double[][] splitsArray;

		if(inputMode == InOutMode.SINGLE){
			inputCols = inputMode.getInputCols(transformer);
			splitsArray = new double[][]{transformer.getSplits()};
		} else

		if(inputMode == InOutMode.MULTIPLE){
			inputCols = inputMode.getInputCols(transformer);
			splitsArray = transformer.getSplitsArray();
		} else

		{
			throw new IllegalArgumentException();
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < inputCols.length; i++){
			String inputCol = inputCols[i];
			double[] splits = splitsArray[i];

			Feature feature = encoder.getOnlyFeature(inputCol);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			Discretize discretize = new Discretize(continuousFeature.getName())
				.setDataType(DataType.INTEGER);

			List<Integer> categories = new ArrayList<>();

			for(int j = 0; j < (splits.length - 1); j++){
				Integer category = j;

				categories.add(category);

				Interval interval = new Interval((j < (splits.length - 2)) ? Interval.Closure.CLOSED_OPEN : Interval.Closure.CLOSED_CLOSED)
					.setLeftMargin(formatMargin(splits[j]))
					.setRightMargin(formatMargin(splits[j + 1]));

				DiscretizeBin discretizeBin = new DiscretizeBin(category, interval);

				discretize.addDiscretizeBins(discretizeBin);
			}

			DerivedField derivedField = encoder.createDerivedField(formatName(transformer, i), OpType.CATEGORICAL, DataType.INTEGER, discretize);

			result.add(new IndexFeature(encoder, derivedField, categories));
		}

		return result;
	}

	static
	private Double formatMargin(double value){

		if(Double.isInfinite(value)){
			return null;
		}

		return value;
	}
}