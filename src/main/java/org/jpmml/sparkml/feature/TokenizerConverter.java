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

import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.feature.Tokenizer;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.OpType;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sparkml.DocumentFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class TokenizerConverter extends FeatureConverter<Tokenizer> {

	public TokenizerConverter(Tokenizer transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeOutputFeatures(SparkMLEncoder encoder){
		Tokenizer transformer = getTransformer();

		Feature feature = encoder.getOnlyFeature(transformer.getInputCol());

		Apply apply = PMMLUtil.createApply("lowercase", feature.ref());

		DerivedField derivedField = encoder.createDerivedField(FeatureUtil.createName("lowercase", feature), OpType.CATEGORICAL, DataType.STRING, apply);

		return Collections.<Feature>singletonList(new DocumentFeature(encoder, derivedField, "\\s+"));
	}
}