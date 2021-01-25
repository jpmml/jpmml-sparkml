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

import java.util.List;
import java.util.Objects;

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
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.IndexFeature;
import org.jpmml.converter.Label;
import org.jpmml.converter.LabelUtil;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
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
		Label label = getLabel(encoder);
		List<Feature> features = getFeatures(encoder);

		Schema result = new Schema(encoder, label, features);

		checkSchema(result);

		return result;
	}

	public Label getLabel(SparkMLEncoder encoder){
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

							List<Integer> categories = LabelUtil.createTargetCategories(numClasses);

							Field<?> field = encoder.toCategorical(continuousFeature.getName(), categories);

							encoder.putOnlyFeature(labelCol, new IndexFeature(encoder, field, categories));

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

			int numClasses = classificationModel.numClasses();

			CategoricalLabel categoricalLabel = (CategoricalLabel)label;

			SchemaUtil.checkSize(numClasses, categoricalLabel);
		}

		return label;
	}

	public List<Feature> getFeatures(SparkMLEncoder encoder){
		T model = getTransformer();

		String featuresCol = model.getFeaturesCol();

		List<Feature> features = encoder.getFeatures(featuresCol);

		if(model instanceof PredictionModel){
			PredictionModel<?, ?> predictionModel = (PredictionModel<?, ?>)model;

			int numFeatures = predictionModel.numFeatures();
			if(numFeatures != -1){
				SchemaUtil.checkSize(numFeatures, features);
			}
		}

		return features;
	}

	public List<OutputField> registerOutputFields(Label label, org.dmg.pmml.Model model, SparkMLEncoder encoder){
		return null;
	}

	public org.dmg.pmml.Model registerModel(SparkMLEncoder encoder){
		Schema schema = encodeSchema(encoder);

		Label label = schema.getLabel();

		org.dmg.pmml.Model model = encodeModel(schema);

		List<OutputField> sparkOutputFields = registerOutputFields(label, model, encoder);
		if(sparkOutputFields != null && sparkOutputFields.size() > 0){
			org.dmg.pmml.Model finalModel = MiningModelUtil.getFinalModel(model);

			Output output = ModelUtil.ensureOutput(finalModel);

			List<OutputField> outputFields = output.getOutputFields();

			outputFields.addAll(sparkOutputFields);
		}

		return model;
	}

	static
	private void checkSchema(Schema schema){
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		if(label == null){
			return;
		}

		for(Feature feature : features){

			if(Objects.equals(label.getName(), feature.getName())){
				throw new IllegalArgumentException("Label column '" + label.getName() + "' is contained in the list of feature columns");
			}
		}
	}
}