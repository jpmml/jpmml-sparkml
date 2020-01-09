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
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class MinMaxScalerModelConverter extends FeatureConverter<MinMaxScalerModel> {

	public MinMaxScalerModelConverter(MinMaxScalerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		MinMaxScalerModel transformer = getTransformer();

		double rescaleFactor = (transformer.getMax() - transformer.getMin());
		double rescaleConstant = transformer.getMin();

		Vector originalMin = transformer.originalMin();
		Vector originalMax = transformer.originalMax();

		List<Feature> features = encoder.getFeatures(transformer.getInputCol());

		SchemaUtil.checkSize(Math.max(originalMin.size(), originalMax.size()), features);

		List<Feature> result = new ArrayList<>();

		for(int i = 0, length = features.size(); i < length; i++){
			Feature feature = features.get(i);

			ContinuousFeature continuousFeature = feature.toContinuousFeature();

			double min = originalMin.apply(i);
			double max = originalMax.apply(i);

			Expression expression = PMMLUtil.createApply(PMMLFunctions.DIVIDE, PMMLUtil.createApply(PMMLFunctions.SUBTRACT, continuousFeature.ref(), PMMLUtil.createConstant(min)), PMMLUtil.createConstant(max - min));

			if(!ValueUtil.isOne(rescaleFactor)){
				expression = PMMLUtil.createApply(PMMLFunctions.MULTIPLY, expression, PMMLUtil.createConstant(rescaleFactor));
			} // End if

			if(!ValueUtil.isZero(rescaleConstant)){
				expression = PMMLUtil.createApply(PMMLFunctions.ADD, expression, PMMLUtil.createConstant(rescaleConstant));
			}

			DerivedField derivedField = encoder.createDerivedField(formatName(transformer, i, length), OpType.CONTINUOUS, DataType.DOUBLE, expression);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}
}