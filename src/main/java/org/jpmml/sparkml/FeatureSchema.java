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

import org.dmg.pmml.FieldName;
import org.jpmml.converter.Schema;

public class FeatureSchema extends Schema {

	private List<Feature> features = null;


	public FeatureSchema(FieldName targetField, List<String> targetCategories, List<FieldName> activeFields, List<Feature> features){
		super(targetField, targetCategories, activeFields);

		setFeatures(features);
	}

	public Feature getFeature(int index){
		List<Feature> features = getFeatures();

		return features.get(index);
	}

	public List<Feature> getFeatures(){
		return this.features;
	}

	private void setFeatures(List<Feature> features){
		this.features = features;
	}
}