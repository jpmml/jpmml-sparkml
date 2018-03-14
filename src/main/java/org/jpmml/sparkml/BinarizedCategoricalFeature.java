/*
 * Copyright (c) 2018 Villu Ruusmann
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

import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;

public class BinarizedCategoricalFeature extends Feature {

	private List<BinaryFeature> binaryFeatures = null;


	public BinarizedCategoricalFeature(PMMLEncoder encoder, FieldName name, DataType dataType, List<BinaryFeature> binaryFeatures){
		super(encoder, name, dataType);

		setBinaryFeatures(binaryFeatures);
	}

	@Override
	public ContinuousFeature toContinuousFeature(){
		throw new UnsupportedOperationException();
	}

	public List<BinaryFeature> getBinaryFeatures(){
		return this.binaryFeatures;
	}

	private void setBinaryFeatures(List<BinaryFeature> binaryFeatures){

		if(binaryFeatures == null || binaryFeatures.size() < 1){
			throw new IllegalArgumentException();
		}

		this.binaryFeatures = binaryFeatures;
	}
}