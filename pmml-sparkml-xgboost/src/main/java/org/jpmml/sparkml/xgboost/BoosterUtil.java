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

import ml.dmlc.xgboost4j.scala.Booster;
import ml.dmlc.xgboost4j.scala.spark.params.GeneralParams;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.ModelConverter;
import org.jpmml.xgboost.HasXGBoostOptions;
import org.jpmml.xgboost.Learner;
import org.jpmml.xgboost.XGBoostUtil;

public class BoosterUtil {

	private BoosterUtil(){
	}

	static
	public <M extends Model<M> & HasPredictionCol & GeneralParams, C extends ModelConverter<M> & HasXGBoostOptions> MiningModel encodeBooster(C converter, Booster booster, Schema schema){
		M model = converter.getModel();

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

		Float missing = model.getMissing();
		if(missing.isNaN()){
			missing = null;
		}

		Map<String, Object> options = new LinkedHashMap<>();
		options.put(HasXGBoostOptions.OPTION_MISSING, converter.getOption(HasXGBoostOptions.OPTION_MISSING, missing));
		options.put(HasXGBoostOptions.OPTION_COMPACT, converter.getOption(HasXGBoostOptions.OPTION_COMPACT, false));
		options.put(HasXGBoostOptions.OPTION_NUMERIC, converter.getOption(HasXGBoostOptions.OPTION_NUMERIC, true));
		options.put(HasXGBoostOptions.OPTION_PRUNE, converter.getOption(HasXGBoostOptions.OPTION_PRUNE, false));
		options.put(HasXGBoostOptions.OPTION_NTREE_LIMIT, converter.getOption(HasXGBoostOptions.OPTION_NTREE_LIMIT, null));

		Boolean numeric = (Boolean)options.get(HasXGBoostOptions.OPTION_NUMERIC);

		Schema xgbSchema = learner.toXGBoostSchema(numeric, schema);

		return learner.encodeMiningModel(options, xgbSchema);
	}
}