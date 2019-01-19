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

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.classification.ClassificationModel;
import org.apache.spark.ml.param.shared.HasFeaturesCol;
import org.apache.spark.ml.param.shared.HasLabelCol;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Field;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;

abstract
public class ModelConverter<T extends Model<T> & HasFeaturesCol & HasPredictionCol> extends TransformerConverter<T> {

	public ModelConverter(T model){
		super(model);
	}

	abstract
	public MiningFunction getMiningFunction();

	abstract
	public org.dmg.pmml.Model encodeModel(Schema schema);

	public Schema encodeSchema(SparkMLEncoder encoder){
		T model = getTransformer();

		Label label = null;

		if(model instanceof HasLabelCol){
			HasLabelCol hasLabelCol = (HasLabelCol)model;

			String labelCol = hasLabelCol.getLabelCol();

			Feature feature = encoder.getOnlyFeature(labelCol);

			MiningFunction miningFunction = getMiningFunction();
			switch(miningFunction){
				case CLASSIFICATION:
					{
						if(feature instanceof BooleanFeature){
							BooleanFeature booleanFeature = (BooleanFeature)feature;

							label = new CategoricalLabel(booleanFeature.getName(), booleanFeature.getDataType(), booleanFeature.getValues());
						} else

						if(feature instanceof CategoricalFeature){
							CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

							DataField dataField = (DataField)categoricalFeature.getField();

							label = new CategoricalLabel(dataField);
						} else

						if(feature instanceof ContinuousFeature){
							ContinuousFeature continuousFeature = (ContinuousFeature)feature;

							int numClasses = 2;

							if(model instanceof ClassificationModel){
								ClassificationModel<?, ?> classificationModel = (ClassificationModel<?, ?>)model;

								numClasses = classificationModel.numClasses();
							}

							List<String> categories = new ArrayList<>();

							for(int i = 0; i < numClasses; i++){
								categories.add(String.valueOf(i));
							}

							Field<?> field = encoder.toCategorical(continuousFeature.getName(), categories);

							encoder.putOnlyFeature(labelCol, new CategoricalFeature(encoder, field, categories));

							label = new CategoricalLabel(field.getName(), field.getDataType(), categories);
						} else

						{
							throw new IllegalArgumentException("Expected a categorical or categorical-like continuous feature, got " + feature);
						}
					}
					break;
				case REGRESSION:
					{
						Field<?> field = encoder.toContinuous(feature.getName());

						field.setDataType(DataType.DOUBLE);

						label = new ContinuousLabel(field.getName(), field.getDataType());
					}
					break;
				default:
					throw new IllegalArgumentException("Mining function " + miningFunction + " is not supported");
			}
		}

		if(model instanceof ClassificationModel){
			ClassificationModel<?, ?> classificationModel = (ClassificationModel<?, ?>)model;

			CategoricalLabel categoricalLabel = (CategoricalLabel)label;

			int numClasses = classificationModel.numClasses();
			if(numClasses != categoricalLabel.size()){
				throw new IllegalArgumentException("Expected " + numClasses + " target categories, got " + categoricalLabel.size() + " target categories");
			}
		}

		String featuresCol = model.getFeaturesCol();

		List<Feature> features = encoder.getFeatures(featuresCol);

		if(model instanceof PredictionModel){
			PredictionModel<?, ?> predictionModel = (PredictionModel<?, ?>)model;

			int numFeatures = predictionModel.numFeatures();
			if(numFeatures != -1 && features.size() != numFeatures){
				throw new IllegalArgumentException("Expected " + numFeatures + " features, got " + features.size() + " features");
			}
		}

		Schema result = new Schema(label, features);

		return result;
	}

	public List<OutputField> registerOutputFields(Label label, SparkMLEncoder encoder){
		return null;
	}

	public org.dmg.pmml.Model registerModel(SparkMLEncoder encoder){
		Schema schema = encodeSchema(encoder);

		Label label = schema.getLabel();

		org.dmg.pmml.Model model = encodeModel(schema);

		List<OutputField> sparkOutputFields = registerOutputFields(label, encoder);
		if(sparkOutputFields != null && sparkOutputFields.size() > 0){
			Output output;

			if(model instanceof MiningModel){
				MiningModel miningModel = (MiningModel)model;

				org.dmg.pmml.Model finalModel = MiningModelUtil.getFinalModel(miningModel);

				output = ModelUtil.ensureOutput(finalModel);
			} else

			{
				output = ModelUtil.ensureOutput(model);
			}

			List<OutputField> outputFields = output.getOutputFields();

			outputFields.addAll(0, sparkOutputFields);
		}

		return model;
	}
}