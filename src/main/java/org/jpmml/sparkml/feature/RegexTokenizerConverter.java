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

import org.apache.spark.ml.feature.RegexTokenizer;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.dmg.pmml.TypeDefinitionField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sparkml.DocumentFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class RegexTokenizerConverter extends FeatureConverter<RegexTokenizer> {

	public RegexTokenizerConverter(RegexTokenizer transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		RegexTokenizer transformer = getTransformer();

		if(!transformer.getGaps()){
			throw new IllegalArgumentException("Expected splitter mode, got token matching mode");
		} // End if

		if(transformer.getMinTokenLength() != 1){
			throw new IllegalArgumentException("Expected 1 as minimum token length, got " + transformer.getMinTokenLength() + " as minimum token length");
		}

		Feature feature = encoder.getOnlyFeature(transformer.getInputCol());

		TypeDefinitionField field = encoder.getField(feature.getName());

		if(transformer.getToLowercase()){
			Apply apply = PMMLUtil.createApply("lowercase", feature.ref());

			field = encoder.createDerivedField(FeatureUtil.createName("lowercase", feature), OpType.CATEGORICAL, DataType.STRING, apply);
		}

		return Collections.<Feature>singletonList(new DocumentFeature(encoder, field, transformer.getPattern()));
	}
}