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
import org.apache.spark.ml.param.shared.HasLabelCol;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Field;
import org.dmg.pmml.MiningFunction;
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.IndexFeature;
import org.jpmml.converter.Label;
import org.jpmml.converter.LabelUtil;
import org.jpmml.converter.SchemaUtil;

abstract
public class PredictionModelConverter<T extends PredictionModel<Vector, T> & HasLabelCol & HasFeaturesCol & HasPredictionCol> extends ModelConverter<T> {

	public PredictionModelConverter(T model){
		super(model);
	}

	@Override
	public Label getLabel(SparkMLEncoder encoder){
		T model = getModel();

		String labelCol = model.getLabelCol();

		Feature feature = encoder.getOnlyFeature(labelCol);

		MiningFunction miningFunction = getMiningFunction();
		switch(miningFunction){
			case CLASSIFICATION:
				{
					if(feature instanceof BooleanFeature){
						BooleanFeature booleanFeature = (BooleanFeature)feature;

						return new CategoricalLabel(booleanFeature);
					} else

					if(feature instanceof CategoricalFeature){
						CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

						DataField dataField = (DataField)categoricalFeature.getField();

						return new CategoricalLabel(dataField);
					} else

					if(feature instanceof ContinuousFeature){
						ContinuousFeature continuousFeature = (ContinuousFeature)feature;

						int numClasses = 2;

						if(this instanceof ClassificationModelConverter){
							ClassificationModelConverter<?> classificationModelConverter = (ClassificationModelConverter<?>)this;

							numClasses = classificationModelConverter.getNumberOfClasses();
						}

						List<Integer> categories = LabelUtil.createTargetCategories(numClasses);

						Field<?> field = encoder.toCategorical(continuousFeature.getName(), categories);

						encoder.putOnlyFeature(labelCol, new IndexFeature(encoder, field, categories));

						return new CategoricalLabel(field.requireName(), field.requireDataType(), categories);
					} else

					{
						throw new IllegalArgumentException("Expected a categorical or categorical-like continuous feature, got " + feature);
					}
				}
			case REGRESSION:
				{
					Field<?> field = encoder.toContinuous(feature.getName());

					field.setDataType(DataType.DOUBLE);

					return new ContinuousLabel(field);
				}
			default:
				throw new IllegalArgumentException("Mining function " + miningFunction + " is not supported");
		}
	}

	@Override
	public List<Feature> getFeatures(SparkMLEncoder encoder){
		T model = getModel();

		String featuresCol = model.getFeaturesCol();

		List<Feature> features = encoder.getFeatures(featuresCol);

		int numFeatures = model.numFeatures();
		if(numFeatures != -1){
			SchemaUtil.checkSize(numFeatures, features);
		}

		return features;
	}
}