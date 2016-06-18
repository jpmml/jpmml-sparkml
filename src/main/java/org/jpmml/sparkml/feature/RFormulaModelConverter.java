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

import java.util.List;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.feature.RFormulaModel;
import org.apache.spark.ml.feature.ResolvedRFormula;
import org.jpmml.converter.Feature;
import org.jpmml.sparkml.ConverterUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;
import org.jpmml.sparkml.TransformerUtil;

public class RFormulaModelConverter extends FeatureConverter<RFormulaModel> {

	public RFormulaModelConverter(RFormulaModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		RFormulaModel transformer = getTransformer();

		ResolvedRFormula resolvedFormula = (ResolvedRFormula)TransformerUtil.getParam(transformer, "resolvedFormula");

		String targetCol = resolvedFormula.label();

		String labelCol = transformer.getLabelCol();
		if(!(targetCol).equals(labelCol) && !featureMapper.hasFeatures(labelCol)){
			List<Feature> features = featureMapper.getFeatures(targetCol);

			featureMapper.putFeatures(labelCol, features);
		}

		PipelineModel pipelineModel = (PipelineModel)TransformerUtil.getParam(transformer, "pipelineModel");

		Transformer[] stages = pipelineModel.stages();
		for(Transformer stage : stages){
			FeatureConverter<?> featureConverter = ConverterUtil.createFeatureConverter(stage);

			featureMapper.append(featureConverter);
		}

		return null;
	}
}