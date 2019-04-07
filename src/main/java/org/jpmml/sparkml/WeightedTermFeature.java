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

import java.util.Objects;

import org.dmg.pmml.Apply;
import org.dmg.pmml.DefineFunction;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.model.ToStringHelper;

public class WeightedTermFeature extends TermFeature {

	private Number weight = null;


	public WeightedTermFeature(PMMLEncoder encoder, DefineFunction defineFunction, Feature feature, String value, Number weight){
		super(encoder, defineFunction, feature, value);

		setWeight(weight);
	}

	@Override
	public Apply createApply(){
		Number weight = getWeight();

		Apply apply = super.createApply()
			.addExpressions(PMMLUtil.createConstant(weight));

		return apply;
	}

	@Override
	public int hashCode(){
		return (31 * super.hashCode()) + Objects.hashCode(this.getWeight());
	}

	@Override
	public boolean equals(Object object){

		if(object instanceof WeightedTermFeature){
			WeightedTermFeature that = (WeightedTermFeature)object;

			return super.equals(object) && Objects.equals(this.getWeight(), that.getWeight());
		}

		return false;
	}

	@Override
	protected ToStringHelper toStringHelper(){
		return super.toStringHelper()
			.add("weight", getWeight());
	}

	public Number getWeight(){
		return this.weight;
	}

	private void setWeight(Number weight){
		this.weight = Objects.requireNonNull(weight);
	}
}