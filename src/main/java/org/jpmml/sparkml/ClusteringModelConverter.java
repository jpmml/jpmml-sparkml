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
package org.jpmml.sparkml;

import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.shared.HasFeaturesCol;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;

abstract
public class ClusteringModelConverter<T extends Model<T> & HasFeaturesCol & HasPredictionCol> extends ModelConverter<T> {

	public ClusteringModelConverter(T model){
		super(model);
	}

	@Override
	public MiningFunction getMiningFunction(){
		return MiningFunction.CLUSTERING;
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, SparkMLEncoder encoder){
		T model = getTransformer();

		String predictionCol = model.getPredictionCol();

		OutputField predictedField = ModelUtil.createPredictedField(FieldName.create(predictionCol), DataType.STRING, OpType.CATEGORICAL);

		Feature feature = new Feature(encoder, predictedField.getName(), predictedField.getDataType()){

			@Override
			public ContinuousFeature toContinuousFeature(){
				throw new UnsupportedOperationException();
			}
		};

		encoder.putOnlyFeature(predictionCol, feature);

		return Collections.singletonList(predictedField);
	}
}