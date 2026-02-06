/*
 * Copyright (c) 2022 Villu Ruusmann
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
package org.jpmml.sparkml.lightgbm;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMModelMethods;
import com.microsoft.azure.synapse.ml.lightgbm.booster.LightGBMBooster;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import org.jpmml.lightgbm.GBDT;
import org.jpmml.lightgbm.HasLightGBMOptions;
import org.jpmml.lightgbm.LightGBMUtil;
import org.jpmml.sparkml.ModelConverter;
import scala.Option;

public class BoosterUtil {

	private BoosterUtil(){
	}

	static
	public <C extends ModelConverter<M>, M extends Model<M> & HasPredictionCol & LightGBMModelMethods> MiningModel encodeModel(C converter, Schema schema){
		M model = converter.getModel();

		GBDT gbdt = BoosterUtil.getGBDT(model);

		Integer bestIteration = model.getBoosterBestIteration();
		if(bestIteration < 0){
			bestIteration = null;
		}

		Map<String, Object> options = new LinkedHashMap<>();
		options.put(HasLightGBMOptions.OPTION_COMPACT, converter.getOption(HasLightGBMOptions.OPTION_COMPACT, Boolean.TRUE));
		options.put(HasLightGBMOptions.OPTION_NUM_ITERATION, converter.getOption(HasLightGBMOptions.OPTION_NUM_ITERATION, bestIteration));

		Schema lgbmSchema = gbdt.toLightGBMSchema(schema);

		MiningModel miningModel = gbdt.encodeModel(options, lgbmSchema);

		return miningModel;
	}

	static
	private <M extends Model<M> & LightGBMModelMethods> GBDT getGBDT(M model){
		LightGBMBooster booster = model.getLightGBMBooster();

		Option<String> modelStr = booster.modelStr();
		if(modelStr.isEmpty()){
			throw new IllegalArgumentException();
		}

		String string = modelStr.get();

		try(BufferedReader reader = new BufferedReader(new StringReader(string))){
			Iterator<String> linesIt = reader.lines()
				.iterator();

			return LightGBMUtil.loadGBDT(linesIt);
		} catch(IOException ioe){
			throw new RuntimeException(ioe);
		}
	}
}