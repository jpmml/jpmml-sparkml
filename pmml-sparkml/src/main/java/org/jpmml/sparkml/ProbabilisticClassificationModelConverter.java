/*
 * Copyright (c) 2023 Villu Ruusmann
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

import org.apache.spark.ml.classification.ProbabilisticClassificationModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.shared.HasProbabilityCol;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Model;
import org.dmg.pmml.OutputField;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;

abstract
public class ProbabilisticClassificationModelConverter<T extends ProbabilisticClassificationModel<Vector, T> & HasProbabilityCol> extends ClassificationModelConverter<T> {

	public ProbabilisticClassificationModelConverter(T model){
		super(model);
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, Model pmmlModel, SparkMLEncoder encoder){
		T model = getModel();

		List<OutputField> result = super.registerOutputFields(label, pmmlModel, encoder);

		CategoricalLabel categoricalLabel = (CategoricalLabel)label;

		String probabilityCol = model.getProbabilityCol();

		result = new ArrayList<>(result);

		List<Feature> features = new ArrayList<>();

		for(int i = 0; i < categoricalLabel.size(); i++){
			Object value = categoricalLabel.getValue(i);

			OutputField probabilityField = ModelUtil.createProbabilityField(FieldNameUtil.create(probabilityCol, value), DataType.DOUBLE, value);

			result.add(probabilityField);

			features.add(new ContinuousFeature(encoder, probabilityField));
		}

		// XXX
		encoder.putFeatures(probabilityCol, features);

		return result;
	}
}