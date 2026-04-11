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
import java.util.function.Supplier;

import org.apache.spark.ml.feature.ElementwiseProduct;
import org.apache.spark.ml.linalg.Vector;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ConstantFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ProductFeature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class ElementwiseProductConverter extends FeatureConverter<ElementwiseProduct> {

	public ElementwiseProductConverter(ElementwiseProduct transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		ElementwiseProduct transformer = getTransformer();

		Vector scalingVec = transformer.getScalingVec();

		List<Feature> features = encoder.getFeatures(transformer.getInputCol());

		SchemaUtil.checkSize(scalingVec.size(), features);

		List<String> names = formatNames(features.size(), encoder);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			String name = names.get(i);

			Number factor = scalingVec.apply(i);

			if(ValueUtil.isZero(factor)){
				feature = new ConstantFeature(encoder, 0d);
			} else

			if(ValueUtil.isOne(factor)){
				// Ignored
			} else

			{
				feature = new ProductFeature(encoder, feature, factor){

					@Override
					public ContinuousFeature toContinuousFeature(){
						Supplier<Apply> applySupplier = () -> {
							Feature feature = getFeature();
							Number factor = getFactor();

							return ExpressionUtil.createApply(PMMLFunctions.MULTIPLY, (feature.toContinuousFeature()).ref(), ExpressionUtil.createConstant(factor));
						};

						return toContinuousFeature(name, DataType.DOUBLE, applySupplier);
					}
				};
			}

			result.add(feature);
		}

		return result;
	}
}