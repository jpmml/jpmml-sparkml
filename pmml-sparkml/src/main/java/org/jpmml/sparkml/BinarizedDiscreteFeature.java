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
import java.util.Objects;

import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.DiscreteFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.model.ToStringHelper;

public class BinarizedDiscreteFeature extends Feature {

	private List<BinaryFeature> binaryFeatures = null;


	public BinarizedDiscreteFeature(PMMLEncoder encoder, DiscreteFeature discreteFeature, List<BinaryFeature> binaryFeatures){
		super(encoder, discreteFeature.getName(), discreteFeature.getDataType());

		setBinaryFeatures(binaryFeatures);
	}

	@Override
	public ContinuousFeature toContinuousFeature(){
		throw new UnsupportedOperationException();
	}

	@Override
	public int hashCode(){
		return (31 * super.hashCode()) + Objects.hashCode(this.getBinaryFeatures());
	}

	@Override
	public boolean equals(Object object){

		if(object instanceof BinarizedDiscreteFeature){
			BinarizedDiscreteFeature that = (BinarizedDiscreteFeature)object;

			return super.equals(that) && Objects.equals(this.getBinaryFeatures(), that.getBinaryFeatures());
		}

		return false;
	}

	@Override
	protected ToStringHelper toStringHelper(){
		return super.toStringHelper()
			.add("binaryFeatures", getBinaryFeatures());
	}

	public List<BinaryFeature> getBinaryFeatures(){
		return this.binaryFeatures;
	}

	private void setBinaryFeatures(List<BinaryFeature> binaryFeatures){

		if(binaryFeatures != null && binaryFeatures.isEmpty()){
			throw new IllegalArgumentException();
		}

		this.binaryFeatures = Objects.requireNonNull(binaryFeatures);
	}
}