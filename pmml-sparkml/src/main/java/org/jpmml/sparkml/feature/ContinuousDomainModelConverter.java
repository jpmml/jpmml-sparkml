/*
 * Copyright (c) 2025 Villu Ruusmann
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
import java.util.Map;

import org.dmg.pmml.DataField;
import org.dmg.pmml.Field;
import org.dmg.pmml.Interval;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.OutlierTreatmentMethod;
import org.dmg.pmml.Value;
import org.jpmml.converter.Feature;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.converter.OutlierDecorator;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class ContinuousDomainModelConverter extends FeatureConverter<ContinuousDomainModel> {

	public ContinuousDomainModelConverter(ContinuousDomainModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		ContinuousDomainModel transformer = getTransformer();

		Object[] missingValues = transformer.getMissingValues();

		MissingValueTreatmentMethod missingValueTreatment = parseMissingValueTreatment(transformer.getMissingValueTreatment());
		Object missingValueReplacement = transformer.getMissingValueReplacement();
		InvalidValueTreatmentMethod invalidValueTreatment = parseInvalidValueTreatment(transformer.getInvalidValueTreatment());
		Object invalidValueReplacement = transformer.getInvalidValueReplacement();

		OutlierTreatmentMethod outlierTreatment = parseOutlierTreatment(transformer.getOutlierTreatment());
		Number lowValue;
		Number highValue;

		switch(outlierTreatment){
			case AS_MISSING_VALUES:
			case AS_EXTREME_VALUES:
				lowValue = transformer.getLowValue();
				highValue = transformer.getHighValue();
				break;
			default:
				lowValue = null;
				highValue = null;
				break;
		}

		boolean withData = transformer.getWithData();

		Map<String, Number[]> dataRanges = Collections.emptyMap();

		if(withData){
			dataRanges = DomainUtil.toJavaMap(transformer.getDataRanges());
		}

		List<Feature> result = new ArrayList<>();

		String[] inputCols = transformer.getInputCols();
		for(String inputCol : inputCols){
			Feature feature = encoder.getOnlyFeature(inputCol);

			Number[] dataRange = dataRanges.get(inputCol);

			Field<?> field = feature.getField();

			if(field instanceof DataField){
				DataField dataField = (DataField)field;

				dataField.addValues(Value.Property.MISSING, missingValues);

				encoder.addDecorator(dataField, new MissingValueDecorator(missingValueTreatment, missingValueReplacement));
				encoder.addDecorator(dataField, new InvalidValueDecorator(invalidValueTreatment, invalidValueReplacement));

				encoder.addDecorator(dataField, new OutlierDecorator(outlierTreatment, lowValue, highValue));

				if(dataRange != null){
					Interval interval = new Interval(Interval.Closure.CLOSED_CLOSED, dataRange[0], dataRange[1]);

					dataField.addIntervals(interval);
				}
			} else

			{
				throw new IllegalArgumentException();
			}

			result.add(feature);
		}

		return result;
	}

	static
	private MissingValueTreatmentMethod parseMissingValueTreatment(String missingValueTreatment){
		return MissingValueTreatmentMethod.fromValue(missingValueTreatment);
	}

	static
	private InvalidValueTreatmentMethod parseInvalidValueTreatment(String invalidValueTreatment){
		return InvalidValueTreatmentMethod.fromValue(invalidValueTreatment);
	}

	static
	private OutlierTreatmentMethod parseOutlierTreatment(String outlierTreatment){
		return OutlierTreatmentMethod.fromValue(outlierTreatment);
	}
}