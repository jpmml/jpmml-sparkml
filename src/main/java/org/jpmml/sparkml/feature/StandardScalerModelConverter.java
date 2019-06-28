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
import java.util.function.Supplier;

import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.linalg.Vector;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ProductFeature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class StandardScalerModelConverter extends FeatureConverter<StandardScalerModel> {

	public StandardScalerModelConverter(StandardScalerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		StandardScalerModel transformer = getTransformer();

		Vector mean = transformer.mean();
		Vector std = transformer.std();

		boolean withMean = transformer.getWithMean();
		boolean withStd = transformer.getWithStd();

		List<Feature> features = encoder.getFeatures(transformer.getInputCol());

		if(withMean){
			SchemaUtil.checkSize(mean.size(), features);
		} // End if

		if(withStd){
			SchemaUtil.checkSize(std.size(), features);
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < features.size(); i++){
			Feature feature = features.get(i);

			FieldName name = formatName(transformer, i);

			Expression expression = null;

			if(withMean){
				double meanValue = mean.apply(i);

				if(!ValueUtil.isZero(meanValue)){
					ContinuousFeature continuousFeature = feature.toContinuousFeature();

					expression = PMMLUtil.createApply(PMMLFunctions.SUBTRACT, continuousFeature.ref(), PMMLUtil.createConstant(meanValue));
				}
			} // End if

			if(withStd){
				double stdValue = std.apply(i);

				if(!ValueUtil.isOne(stdValue)){
					Double factor = (1d / stdValue);

					if(expression != null){
						expression = PMMLUtil.createApply(PMMLFunctions.MULTIPLY, expression, PMMLUtil.createConstant(factor));
					} else

					{
						feature = new ProductFeature(encoder, feature, factor){

							@Override
							public ContinuousFeature toContinuousFeature(){
								Supplier<Apply> applySupplier = () -> {
									Feature feature = getFeature();
									Number factor = getFactor();

									return PMMLUtil.createApply(PMMLFunctions.MULTIPLY, (feature.toContinuousFeature()).ref(), PMMLUtil.createConstant(factor));
								};

								return toContinuousFeature(name, DataType.DOUBLE, applySupplier);
							}
						};
					}
				}
			} // End if

			if(expression != null){
				DerivedField derivedField = encoder.createDerivedField(name, OpType.CONTINUOUS, DataType.DOUBLE, expression);

				result.add(new ContinuousFeature(encoder, derivedField));
			} else

			{
				result.add(feature);
			}
		}

		return result;
	}
}