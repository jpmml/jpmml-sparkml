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

import org.apache.spark.ml.feature.CountVectorizerModel;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.ParameterField;
import org.dmg.pmml.TextIndex;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sparkml.DocumentFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;
import org.jpmml.sparkml.TermUtil;

public class CountVectorizerModelConverter extends FeatureConverter<CountVectorizerModel> {

	public CountVectorizerModelConverter(CountVectorizerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		CountVectorizerModel transformer = getTransformer();

		DocumentFeature documentFeature = (DocumentFeature)encoder.getOnlyFeature(transformer.getInputCol());

		ParameterField documentField = new ParameterField(FieldName.create("document"));

		ParameterField termField = new ParameterField(FieldName.create("term"));

		TextIndex textIndex = new TextIndex(documentField.getName())
			.setTokenize(Boolean.TRUE)
			.setWordSeparatorCharacterRE(documentFeature.getWordSeparatorRE())
			.setLocalTermWeights(transformer.getBinary() ? TextIndex.LocalTermWeights.BINARY : null)
			.setExpression(new FieldRef(termField.getName()));

		DefineFunction defineFunction = new DefineFunction("tf", OpType.CONTINUOUS, null)
			.setDataType(DataType.INTEGER)
			.addParameterFields(documentField, termField)
			.setExpression(textIndex);

		encoder.addDefineFunction(defineFunction);

		List<Feature> result = new ArrayList<>();

		String[] vocabulary = transformer.vocabulary();
		for(int i = 0; i < vocabulary.length; i++){
			String term = vocabulary[i];
			String trimmedTerm = TermUtil.trim(term);

			if(!(term).equals(trimmedTerm)){
				throw new IllegalArgumentException(term);
			}

			Constant constant = PMMLUtil.createConstant(term)
				.setDataType(DataType.STRING);

			final
			Apply apply = PMMLUtil.createApply(defineFunction.getName(), documentFeature.ref(), constant);

			Feature termFeature = new Feature(encoder, FieldName.create(defineFunction.getName() + "(" + term + ")"), DataType.INTEGER){

				@Override
				public ContinuousFeature toContinuousFeature(){
					PMMLEncoder encoder = ensureEncoder();

					DerivedField derivedField = encoder.getDerivedField(getName());
					if(derivedField == null){
						derivedField = encoder.createDerivedField(getName(), OpType.CONTINUOUS, getDataType(), apply);
					}

					return new ContinuousFeature(encoder, derivedField);
				}
			};

			result.add(termFeature);
		}

		return result;
	}
}