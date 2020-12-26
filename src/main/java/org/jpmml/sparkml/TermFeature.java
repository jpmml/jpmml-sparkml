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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DefineFunction;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.ParameterField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.PMMLEncoder;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.model.ToStringHelper;

public class TermFeature extends Feature {

	private DefineFunction defineFunction = null;

	private Feature feature = null;

	private String value = null;


	public TermFeature(PMMLEncoder encoder, DefineFunction defineFunction, Feature feature, String value){
		super(encoder, FieldNameUtil.create(defineFunction.getName(), value), defineFunction.getDataType());

		setDefineFunction(defineFunction);

		setFeature(feature);
		setValue(value);
	}

	@Override
	public ContinuousFeature toContinuousFeature(){
		return toContinuousFeature(getName(), getDataType(), () -> createApply());
	}

	public WeightedTermFeature toWeightedTermFeature(Number weight){
		PMMLEncoder encoder = getEncoder();

		DefineFunction defineFunction = getDefineFunction();

		String name = (defineFunction.getName()).replace("tf@", "tf-idf@");

		DefineFunction weightedDefineFunction = encoder.getDefineFunction(name);
		if(weightedDefineFunction == null){
			ParameterField weightField = new ParameterField(FieldName.create("weight"));

			List<ParameterField> parameterFields = new ArrayList<>(defineFunction.getParameterFields());
			parameterFields.add(weightField);

			Apply apply = PMMLUtil.createApply(PMMLFunctions.MULTIPLY, defineFunction.getExpression(), new FieldRef(weightField.getName()));

			weightedDefineFunction = new DefineFunction(name, OpType.CONTINUOUS, DataType.DOUBLE, parameterFields, apply);

			encoder.addDefineFunction(weightedDefineFunction);
		}

		return new WeightedTermFeature(encoder, weightedDefineFunction, getFeature(), getValue(), weight);
	}

	public Apply createApply(){
		DefineFunction defineFunction = getDefineFunction();
		Feature feature = getFeature();
		String value = getValue();

		Constant constant = PMMLUtil.createConstant(value, DataType.STRING);

		return PMMLUtil.createApply(defineFunction.getName(), feature.ref(), constant);
	}

	@Override
	public int hashCode(){
		int result = super.hashCode();

		result = (31 * result) + Objects.hashCode(this.getDefineFunction());
		result = (31 * result) + Objects.hashCode(this.getFeature());
		result = (31 * result) + Objects.hashCode(this.getValue());

		return result;
	}

	@Override
	public boolean equals(Object object){

		if(object instanceof TermFeature){
			TermFeature that = (TermFeature)object;

			return super.equals(object) && Objects.equals(this.getDefineFunction(), that.getDefineFunction()) && Objects.equals(this.getFeature(), that.getFeature()) && Objects.equals(this.getValue(), that.getValue());
		}

		return false;
	}

	@Override
	protected ToStringHelper toStringHelper(){
		return super.toStringHelper()
			.add("defineFunction", getDefineFunction())
			.add("feature", getFeature())
			.add("value", getValue());
	}

	public DefineFunction getDefineFunction(){
		return this.defineFunction;
	}

	private void setDefineFunction(DefineFunction defineFunction){
		this.defineFunction = Objects.requireNonNull(defineFunction);
	}

	public Feature getFeature(){
		return this.feature;
	}

	private void setFeature(Feature feature){
		this.feature = Objects.requireNonNull(feature);
	}

	public String getValue(){
		return this.value;
	}

	private void setValue(String value){
		this.value = Objects.requireNonNull(value);
	}
}