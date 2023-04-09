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
package org.jpmml.sparkml.xgboost;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;

import ml.dmlc.xgboost4j.scala.Booster;
import org.dmg.pmml.DataType;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.ModelConverter;
import org.jpmml.xgboost.HasXGBoostOptions;
import org.jpmml.xgboost.Learner;
import org.jpmml.xgboost.XGBoostUtil;

public class BoosterUtil {

	private BoosterUtil(){
	}

	static
	public <C extends ModelConverter<?> & HasXGBoostOptions> MiningModel encodeBooster(C converter, Booster booster, Schema schema){
		byte[] bytes;

		try {
			bytes = booster.toByteArray();
		} catch(Exception e){
			throw new RuntimeException(e);
		}

		Learner learner;

		try(InputStream is = new ByteArrayInputStream(bytes)){
			learner = XGBoostUtil.loadLearner(is);
		} catch(IOException ioe){
			throw new RuntimeException(ioe);
		}

		Function<Feature, Feature> function = new Function<Feature, Feature>(){

			@Override
			public Feature apply(Feature feature){

				if(feature instanceof BinaryFeature){
					BinaryFeature binaryFeature = (BinaryFeature)feature;

					return binaryFeature;
				} else

				{
					ContinuousFeature continuousFeature = feature.toContinuousFeature(DataType.FLOAT);

					return continuousFeature;
				}
			}
		};

		Map<String, Object> options = new LinkedHashMap<>();
		options.put(HasXGBoostOptions.OPTION_COMPACT, converter.getOption(HasXGBoostOptions.OPTION_COMPACT, false));
		options.put(HasXGBoostOptions.OPTION_NTREE_LIMIT, converter.getOption(HasXGBoostOptions.OPTION_NTREE_LIMIT, null));

		Schema xgbSchema = schema.toTransformedSchema(function);

		return learner.encodeMiningModel(options, xgbSchema);
	}
}