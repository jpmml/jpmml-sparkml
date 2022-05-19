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

import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.linalg.DenseMatrix;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.MatrixUtil;
import org.jpmml.sparkml.SparkMLEncoder;

public class PCAModelConverter extends FeatureConverter<PCAModel> {

	public PCAModelConverter(PCAModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		PCAModel transformer = getTransformer();

		DenseMatrix pc = transformer.pc();

		List<Feature> features = encoder.getFeatures(transformer.getInputCol());

		MatrixUtil.checkRows(features.size(), pc);

		List<Feature> result = new ArrayList<>();

		for(int i = 0, length = transformer.getK(); i < length; i++){
			Apply apply = PMMLUtil.createApply(PMMLFunctions.SUM);

			for(int j = 0; j < features.size(); j++){
				Feature feature = features.get(j);

				ContinuousFeature continuousFeature = feature.toContinuousFeature();

				Expression expression = continuousFeature.ref();

				Double coefficient = pc.apply(j, i);
				if(!ValueUtil.isOne(coefficient)){
					expression = PMMLUtil.createApply(PMMLFunctions.MULTIPLY, expression, PMMLUtil.createConstant(coefficient));
				}

				apply.addExpressions(expression);
			}

			DerivedField derivedField = encoder.createDerivedField(formatName(transformer, i, length), OpType.CONTINUOUS, DataType.DOUBLE, apply);

			result.add(new ContinuousFeature(encoder, derivedField));
		}

		return result;
	}
}