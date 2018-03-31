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
import org.jpmml.converter.Feature;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.ValueUtil;
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

		String[] inputCols = transformer.getInputCols();
		String[] outputCols = transformer.getOutputCols();
		if(inputCols.length != outputCols.length){
			throw new IllegalArgumentException();
		}

		MissingValueTreatmentMethod missingValueTreatmentMethod = parseStrategy(strategy);

		List<Row> surrogateRows = surrogateDF.collectAsList();
		if(surrogateRows.size() != 1){
			throw new IllegalArgumentException();
		}

		Row surrogateRow = surrogateRows.get(0);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < inputCols.length; i++){
			String inputCol = inputCols[i];
			String outputCol = outputCols[i];

			Feature feature = encoder.getOnlyFeature(inputCol);

			Field<?> field = encoder.getField(feature.getName());

			if(field instanceof DataField){
				DataField dataField = (DataField)field;

				Object surrogate = surrogateRow.getAs(inputCol);

				MissingValueDecorator missingValueDecorator = new MissingValueDecorator()
					.setMissingValueReplacement(ValueUtil.formatValue(surrogate))
					.setMissingValueTreatment(missingValueTreatmentMethod);

				if(missingValue != null && !missingValue.isNaN()){
					missingValueDecorator.addValues(ValueUtil.formatValue(missingValue));
				}

				encoder.addDecorator(feature.getName(), missingValueDecorator);
			} else

			{
				throw new IllegalArgumentException();
			}

			result.add(feature);
		}

		return result;
	}

	@Override
	public void registerFeatures(SparkMLEncoder encoder){
		ImputerModel transformer = getTransformer();

		List<Feature> features = encodeFeatures(encoder);

		String[] outputCols = transformer.getOutputCols();
		if(outputCols.length != features.size()){
			throw new IllegalArgumentException();
		}

		for(int i = 0; i < features.size(); i++){
			String outputCol = outputCols[i];
			Feature feature = features.get(i);

			encoder.putFeatures(outputCol, Collections.singletonList(feature));
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