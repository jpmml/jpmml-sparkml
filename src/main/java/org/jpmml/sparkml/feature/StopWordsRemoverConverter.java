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
import java.util.regex.Pattern;

import org.apache.spark.ml.feature.StopWordsRemover;
import org.jpmml.converter.Feature;
import org.jpmml.sparkml.DocumentFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;
import org.jpmml.sparkml.TermUtil;

public class StopWordsRemoverConverter extends FeatureConverter<StopWordsRemover> {

	public StopWordsRemoverConverter(StopWordsRemover transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		StopWordsRemover transformer = getTransformer();

		DocumentFeature documentFeature = (DocumentFeature)encoder.getOnlyFeature(transformer.getInputCol());

		Pattern pattern = Pattern.compile(documentFeature.getWordSeparatorRE());

		DocumentFeature.StopWordSet stopWordSet = new DocumentFeature.StopWordSet(transformer.getCaseSensitive());

		String[] stopWords = transformer.getStopWords();
		for(String stopWord : stopWords){
			String[] stopTokens = pattern.split(stopWord);

			// Skip multi-token stopwords. See https://issues.apache.org/jira/browse/SPARK-18374
			if(stopTokens.length > 1){
				continue;
			} // End if

			if(TermUtil.hasPunctuation(stopWord)){
				throw new IllegalArgumentException("Punctuated stop words (" + stopWord + ") are not supported");
			}

			stopWordSet.add(stopWord);
		}

		documentFeature.addStopWordSet(stopWordSet);

		return Collections.singletonList(documentFeature);
	}
}