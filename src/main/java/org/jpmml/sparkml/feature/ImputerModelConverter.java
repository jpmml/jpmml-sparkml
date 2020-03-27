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
package org.jpmml.sparkml.feature;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.dmg.pmml.DataField;
import org.dmg.pmml.Field;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.Value;
import org.jpmml.converter.Feature;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class ImputerModelConverter extends FeatureConverter<ImputerModel> {

	public ImputerModelConverter(ImputerModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		ImputerModel transformer = getTransformer();

		Double missingValue = transformer.getMissingValue();
		String strategy = transformer.getStrategy();
		Dataset<Row> surrogateDF = transformer.surrogateDF();

		MissingValueTreatmentMethod missingValueTreatmentMethod = parseStrategy(strategy);

		List<Row> surrogateRows = surrogateDF.collectAsList();
		if(surrogateRows.size() != 1){
			throw new IllegalArgumentException();
		}

		Row surrogateRow = surrogateRows.get(0);

		String[] inputCols;

		if(transformer.isSet(transformer.inputCol())){
			inputCols = new String[]{transformer.getInputCol()};
		} else

		if(transformer.isSet(transformer.inputCols())){
			inputCols = transformer.getInputCols();
		} else

		{
			throw new IllegalArgumentException();
		}

		List<Feature> result = new ArrayList<>();

		for(String inputCol : inputCols){
			Feature feature = encoder.getOnlyFeature(inputCol);

			Field<?> field = feature.getField();

			if(field instanceof DataField){
				DataField dataField = (DataField)field;

				Object surrogate = surrogateRow.getAs(inputCol);

				encoder.addDecorator(dataField, new MissingValueDecorator(missingValueTreatmentMethod, surrogate));

				if(missingValue != null && !missingValue.isNaN()){
					PMMLUtil.addValues(dataField, Collections.singletonList(missingValue), Value.Property.MISSING);
				}
			} else

			{
				throw new IllegalArgumentException();
			}

			result.add(feature);
		}

		return result;
	}

	@Override
	protected OutputMode getOutputMode(){
		ImputerModel transformer = getTransformer();

		if(transformer.isSet(transformer.inputCol())){
			return OutputMode.SINGLE;
		} else

		if(transformer.isSet(transformer.inputCols())){
			return OutputMode.MULTIPLE;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	public MissingValueTreatmentMethod parseStrategy(String strategy){

		switch(strategy){
			case "mean":
				return MissingValueTreatmentMethod.AS_MEAN;
			case "median":
				return MissingValueTreatmentMethod.AS_MEDIAN;
			default:
				throw new IllegalArgumentException(strategy);
		}
	}
}