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

import java.util.Objects;

import com.google.common.base.Objects.ToStringHelper;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FeatureUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.ValueUtil;

abstract
public class ScaledFeature extends Feature {

	private Feature feature = null;

	private Double factor = null;


	public ScaledFeature(PMMLEncoder encoder, Feature feature, Double factor){
		super(encoder, FeatureUtil.createName("scale", feature), ValueUtil.getDataType(factor));

		setFeature(feature);
		setFactor(factor);
	}

	@Override
	public int hashCode(){
		int result = super.hashCode();

		result = (31 * result) + Objects.hashCode(this.getFeature());
		result = (31 * result) + Objects.hashCode(this.getFactor());

		return result;
	}

	@Override
	public boolean equals(Object object){

		if(object instanceof ScaledFeature){
			ScaledFeature that = (ScaledFeature)object;

			return super.equals(object) && Objects.equals(this.getFeature(), that.getFeature()) && Objects.equals(this.getFactor(), that.getFactor());
		}

		return false;
	}

	@Override
	protected ToStringHelper toStringHelper(){
		return super.toStringHelper()
			.add("feature", getFeature())
			.add("factor", getFactor());
	}

	public Feature getFeature(){
		return this.feature;
	}

	private void setFeature(Feature feature){

		if(feature == null){
			throw new IllegalArgumentException();
		}

		this.feature = feature;
	}

	public Double getFactor(){
		return this.factor;
	}

	private void setFactor(Double factor){

		if(factor == null){
			throw new IllegalArgumentException();
		}

		this.factor = factor;
	}
}