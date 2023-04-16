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
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.ScalarLabel;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;

abstract
public class ModelConverter<T extends Model<T> & HasPredictionCol> extends TransformerConverter<T> {

	public ModelConverter(T model){
		super(model);
	}

	abstract
	public MiningFunction getMiningFunction();

	abstract
	public List<Feature> getFeatures(SparkMLEncoder encoder);

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
		return null;
	}

	public void checkSchema(Schema schema){
		Label label = schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		MiningFunction miningFunction = getMiningFunction();
		switch(miningFunction){
			case ASSOCIATION_RULES:
			case CLUSTERING:
				if(label != null){
					throw new IllegalArgumentException("Expected no label, got " + label);
				}
				break;
			case CLASSIFICATION:
			case REGRESSION:
				if(label == null){
					throw new IllegalArgumentException("Expected a label, got no label");
				}
				break;
			default:
				break;
		}

		if(label instanceof ScalarLabel){
			ScalarLabel scalarLabel = (ScalarLabel)label;

			for(Feature feature : features){

				if(Objects.equals(scalarLabel.getName(), feature.getName())){
					throw new IllegalArgumentException("Label column '" + scalarLabel.getName() + "' is contained in the list of feature columns");
				}
			}
		}
	}

	public List<OutputField> registerOutputFields(Label label, org.dmg.pmml.Model model, SparkMLEncoder encoder){
		return null;
	}

	public org.dmg.pmml.Model registerModel(SparkMLEncoder encoder){
		Schema schema = encodeSchema(encoder);

		Label label = schema.getLabel();

		org.dmg.pmml.Model model = encodeModel(schema);

		List<OutputField> sparkOutputFields = registerOutputFields(label, model, encoder);
		if(sparkOutputFields != null && !sparkOutputFields.isEmpty()){
			org.dmg.pmml.Model finalModel = MiningModelUtil.getFinalModel(model);

			Output output = ModelUtil.ensureOutput(finalModel);

			List<OutputField> outputFields = output.getOutputFields();

			outputFields.addAll(sparkOutputFields);
		}

		return model;
	}
}