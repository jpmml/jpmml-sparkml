/*
 * Copyright (c) 2021 Villu Ruusmann
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

import java.util.List;

import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.shared.HasFeaturesCol;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.jpmml.converter.Feature;
import org.jpmml.converter.SchemaUtil;

abstract
public class PredictionModelConverter<T extends PredictionModel<Vector, T> & HasFeaturesCol & HasPredictionCol> extends ModelConverter<T> {

	public PredictionModelConverter(T model){
		super(model);
	}

	@Override
	public List<Feature> getFeatures(SparkMLEncoder encoder){
		T model = getTransformer();

		String featuresCol = model.getFeaturesCol();

		List<Feature> features = encoder.getFeatures(featuresCol);

		int numFeatures = model.numFeatures();
		if(numFeatures != -1){
			SchemaUtil.checkSize(numFeatures, features);
		}

		return features;
	}
}