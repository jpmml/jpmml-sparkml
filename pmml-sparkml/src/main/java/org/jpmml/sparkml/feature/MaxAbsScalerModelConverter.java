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

import org.apache.spark.ml.feature.MaxAbsScalerModel;
import org.apache.spark.ml.linalg.Vector;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ExpressionUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class MaxAbsScalerModelConverter extends FeatureConverter<MaxAbsScalerModel> {

	public MaxAbsScalerModelConverter(MaxAbsScalerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		MaxAbsScalerModel transformer = getTransformer();

		Vector maxAbs = transformer.maxAbs();

		List<Feature> features = encoder.getFeatures(transformer.getInputCol());

		SchemaUtil.checkSize(maxAbs.size(), features);

		List<String> names = formatNames(features.size(), encoder);

		List<Feature> result = new ArrayList<>();

		for(int i = 0, length = features.size(); i < length; i++){
			Feature feature = features.get(i);

			double maxAbsUnzero = maxAbs.apply(i);
			if(maxAbsUnzero == 0d){
				maxAbsUnzero = 1d;
			} // End if

			if(!ValueUtil.isOne(maxAbsUnzero)){
				ContinuousFeature continuousFeature = feature.toContinuousFeature();

				Expression expression = ExpressionUtil.createApply(PMMLFunctions.DIVIDE, continuousFeature.ref(), ExpressionUtil.createConstant(maxAbsUnzero));

				DerivedField derivedField = encoder.createDerivedField(names.get(i), OpType.CONTINUOUS, DataType.DOUBLE, expression);

				feature = new ContinuousFeature(encoder, derivedField);
			}

			result.add(feature);
		}

		return result;
	}
}