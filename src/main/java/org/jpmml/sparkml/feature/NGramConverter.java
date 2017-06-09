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

import org.apache.spark.ml.feature.NGram;
import org.jpmml.converter.Feature;
import org.jpmml.sparkml.DocumentFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class NGramConverter extends FeatureConverter<NGram> {

	public NGramConverter(NGram transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeOutputFeatures(SparkMLEncoder encoder){
		NGram transformer = getTransformer();

		DocumentFeature documentFeature = (DocumentFeature)encoder.getOnlyFeature(transformer.getInputCol());

		return Collections.<Feature>singletonList(documentFeature);
	}
}