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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.feature.Binarizer;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ListFeature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;

public class BinarizerConverter extends FeatureConverter<Binarizer> {

	public BinarizerConverter(Binarizer transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		Binarizer transformer = getTransformer();

		ContinuousFeature inputFeature = (ContinuousFeature)featureMapper.getOnlyFeature(transformer.getInputCol());

		double threshold = transformer.getThreshold();

		Apply apply = new Apply("if")
			.addExpressions(PMMLUtil.createApply("lessOrEqual", inputFeature.ref(), PMMLUtil.createConstant(threshold)))
			.addExpressions(PMMLUtil.createConstant(0d), PMMLUtil.createConstant(1d));

		DerivedField derivedField = featureMapper.createDerivedField(formatName(transformer), OpType.CONTINUOUS, DataType.DOUBLE, apply);

		Feature feature = new ListFeature(derivedField, Arrays.asList("0", "1"));

		return Collections.singletonList(feature);
	}
}