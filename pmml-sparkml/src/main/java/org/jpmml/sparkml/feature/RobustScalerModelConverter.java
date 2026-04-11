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
import java.util.List;

import org.apache.spark.ml.feature.RobustScalerModel;
import org.apache.spark.ml.linalg.Vector;
import org.jpmml.converter.ConstantFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class RobustScalerModelConverter extends FeatureConverter<RobustScalerModel> {

	public RobustScalerModelConverter(RobustScalerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		RobustScalerModel transformer = getTransformer();

		Vector median = transformer.median();
		Vector range = transformer.range();

		boolean withCentering = transformer.getWithCentering();
		boolean withScaling = transformer.getWithScaling();

		List<Feature> features = encoder.getFeatures(transformer.getInputCol());

		if(withCentering){
			SchemaUtil.checkSize(median.size(), features);
		} // End if

		if(withScaling){
			SchemaUtil.checkSize(range.size(), features);
		}

		List<String> names = formatNames(features.size(), encoder);

		List<Feature> result = new ArrayList<>();

		for(int i = 0, length = features.size(); i < length; i++){
			Feature feature = features.get(i);

			String name = names.get(i);

			Number medianValue = (withCentering ? median.apply(i) : null);
			Number rangeValue = (withScaling ? range.apply(i) : null);

			if(rangeValue != null && ValueUtil.isZero(rangeValue)){
				feature = new ConstantFeature(encoder, 0d);
			} else

			{
				feature = StandardScalerModelConverter.encodeTransformation(feature, name, medianValue, rangeValue, encoder);
			}

			result.add(feature);
		}

		return result;
	}
}