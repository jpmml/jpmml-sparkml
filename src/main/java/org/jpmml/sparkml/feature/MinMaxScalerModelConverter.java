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

import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.linalg.Vector;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;

public class MinMaxScalerModelConverter extends FeatureConverter<MinMaxScalerModel> {

	public MinMaxScalerModelConverter(MinMaxScalerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		MinMaxScalerModel transformer = getTransformer();

		double rescaleFactor = (transformer.getMax() - transformer.getMin());
		double rescaleConstant = transformer.getMin();

		List<Feature> inputFeatures = featureMapper.getFeatures(transformer.getInputCol());

		Vector originalMax = transformer.originalMax();
		if(originalMax.size() != inputFeatures.size()){
			throw new IllegalArgumentException();
		}

		Vector originalMin = transformer.originalMin();
		if(originalMin.size() != inputFeatures.size()){
			throw new IllegalArgumentException();
		}

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < inputFeatures.size(); i++){
			ContinuousFeature inputFeature = (ContinuousFeature)inputFeatures.get(i);

			double max = originalMax.apply(i);
			double min = originalMin.apply(i);

			Expression expression = PMMLUtil.createApply("/", PMMLUtil.createApply("-", inputFeature.ref(), PMMLUtil.createConstant(min)), PMMLUtil.createConstant(max - min));

			if(!ValueUtil.isOne(rescaleFactor)){
				expression = PMMLUtil.createApply("*", expression, PMMLUtil.createConstant(rescaleFactor));
			} // End if

			if(!ValueUtil.isZero(rescaleConstant)){
				expression = PMMLUtil.createApply("+", expression, PMMLUtil.createConstant(rescaleConstant));
			}

			DerivedField derivedField = featureMapper.createDerivedField(formatName(transformer, i), OpType.CONTINUOUS, DataType.DOUBLE, expression);

			Feature feature = new ContinuousFeature(derivedField);

			result.add(feature);
		}

		return result;
	}
}