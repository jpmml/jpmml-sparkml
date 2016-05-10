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
package org.jpmml.sparkml;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Visitor;
import org.jpmml.model.visitors.DictionaryCleaner;
import org.jpmml.model.visitors.MiningSchemaCleaner;

public class PipelineModelUtil {

	private PipelineModelUtil(){
	}

	static
	public PMML toPMML(StructType schema, PipelineModel pipelineModel){
		FeatureMapper featureMapper = new FeatureMapper(schema);

		Transformer[] transformers = pipelineModel.stages();
		for(Transformer transformer : transformers){
			TransformerConverter converter;

			try {
				converter = ConverterUtil.createConverter(transformer);
			} catch(Exception e){
				throw new IllegalArgumentException(e);
			}

			if(converter instanceof FeatureConverter){
				FeatureConverter featureConverter = (FeatureConverter)converter;

				featureMapper.append(featureConverter);
			} else

			if(converter instanceof ModelConverter){
				ModelConverter modelConverter = (ModelConverter)converter;

				PredictionModel<?, ?> predictionModel = (PredictionModel<?, ?>)transformer;

				FeatureSchema featureSchema = featureMapper.createSchema(predictionModel);

				Model model = modelConverter.encodeModel(featureSchema);

				PMML pmml = featureMapper.encodePMML()
					.addModels(model);

				List<? extends Visitor> visitors = Arrays.asList(new MiningSchemaCleaner(), new DictionaryCleaner());
				for(Visitor visitor : visitors){
					visitor.applyTo(pmml);
				}

				return pmml;
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		throw new IllegalArgumentException();
	}
}