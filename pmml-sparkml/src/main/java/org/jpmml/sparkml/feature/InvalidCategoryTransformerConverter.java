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
package org.jpmml.sparkml.feature;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.google.common.collect.ListMultimap;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.Field;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.Model;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.Decorator;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.sparkml.MultiFeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class InvalidCategoryTransformerConverter extends MultiFeatureConverter<InvalidCategoryTransformer> {

	public InvalidCategoryTransformerConverter(InvalidCategoryTransformer transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		InvalidCategoryTransformer transformer = getTransformer();

		InOutMode inputMode = getInputMode();

		List<Feature> result = new ArrayList<>();

		String[] inputCols = inputMode.getInputCols(transformer);
		for(String inputCol : inputCols){
			Feature feature = encoder.getOnlyFeature(inputCol);

			if(!(feature instanceof CategoricalFeature)){
				throw new IllegalArgumentException();
			}

			CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

			Field<?> field = categoricalFeature.getField();
			List<?> values = categoricalFeature.getValues();

			Object invalidCategory;

			if(!values.isEmpty()){
				invalidCategory = values.get(values.size() - 1);
			} else

			{
				throw new IllegalArgumentException();
			} // End if

			if(Objects.equals(invalidCategory, "-999") || Objects.equals(invalidCategory, "__unknown")){
				values = values.subList(0, values.size() - 1);
			} else

			{
				throw new IllegalArgumentException();
			} // End if

			if(field instanceof DataField){
				DataField dataField = (DataField)field;

				replaceDecorator(dataField, new InvalidValueDecorator(InvalidValueTreatmentMethod.AS_MISSING, null), encoder);
			} else

			if(field instanceof DerivedField){
				DerivedField derivedField = (DerivedField)field;

				Apply apply = (Apply)derivedField.getExpression();

				List<Expression> expressions = apply.getExpressions();

				if(!expressions.isEmpty()){
					Constant constant = (Constant)expressions.remove(expressions.size() - 1);

					if(!Objects.equals(invalidCategory, constant.getValue())){
						throw new IllegalArgumentException();
					}
				} else

				{
					throw new IllegalArgumentException();
				}
			} else

			{
				throw new IllegalArgumentException();
			}

			result.add(new CategoricalFeature(encoder, field, values));
		}

		return result;
	}

	static
	private void replaceDecorator(Field<?> field, Decorator decorator, SparkMLEncoder encoder){
		Map<Model, ListMultimap<String, Decorator>> modelDecorators = encoder.getDecorators();

		ListMultimap<String, Decorator> decorators = modelDecorators.get(null);
		if(decorators != null){
			List<Decorator> fieldDecorators = decorators.get(field.requireName());

			if(fieldDecorators != null && !fieldDecorators.isEmpty()){

				for(Iterator<Decorator> it = fieldDecorators.iterator(); it.hasNext(); ){
					Decorator fieldDecorator = it.next();

					if(Objects.equals(fieldDecorator.getClass(), decorator.getClass())){
						it.remove();
					}
				}
			}
		}

		decorators.put(field.requireName(), decorator);
	}
}