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

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;

import com.google.common.io.MoreFiles;
import ml.dmlc.xgboost4j.scala.Booster;
import ml.dmlc.xgboost4j.scala.spark.params.GeneralParams;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Field;
import org.dmg.pmml.mining.MiningModel;
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
	public <M extends Model<M> & HasPredictionCol & GeneralParams, C extends ModelConverter<M> & HasSparkMLXGBoostOptions> MiningModel encodeBooster(C converter, Booster booster, Schema schema){
		M model = converter.getModel();

		Learner learner;

		try {
			File tmpBoosterFile = File.createTempFile("Booster", ".json");

			booster.saveModel(tmpBoosterFile.getAbsolutePath());

			try(InputStream is = new FileInputStream(tmpBoosterFile)){
				learner = XGBoostUtil.loadLearner(is);
			}

			MoreFiles.deleteRecursively(tmpBoosterFile.toPath());
		} catch(Exception e){
			throw new RuntimeException(e);
		}

		Boolean inputFloat = (Boolean)converter.getOption(HasSparkMLXGBoostOptions.OPTION_INPUT_FLOAT, null);
		if((Boolean.TRUE).equals(inputFloat)){
			Function<Feature, Feature> function = new Function<Feature, Feature>(){

				@Override
				public Feature apply(Feature feature){

					if(feature instanceof ContinuousFeature){
						ContinuousFeature continuousFeature = (ContinuousFeature)feature;

						DataType dataType = continuousFeature.getDataType();
						switch(dataType){
							case INTEGER:
							case FLOAT:
								break;
							case DOUBLE:
								{
									Field<?> field = continuousFeature.getField();

									field.setDataType(DataType.FLOAT);

									return new ContinuousFeature(continuousFeature.getEncoder(), field);
								}
							default:
								break;
						}
					}

					return feature;
				}
			};

			schema = schema.toTransformedSchema(function);
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