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

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.dmg.pmml.DataField;
import org.dmg.pmml.Interval;
import org.dmg.pmml.OutlierTreatmentMethod;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.OutlierDecorator;
import org.jpmml.sparkml.SparkMLEncoder;

public class ContinuousDomainModelConverter extends DomainModelConverter<ContinuousDomainModel> {

	public ContinuousDomainModelConverter(ContinuousDomainModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		ContinuousDomainModel transformer = getTransformer();

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

		Map<String, Number[]> dataRanges;

		if(withData){
			dataRanges = DomainUtil.toJavaMap(transformer.getDataRanges());
		} else

		{
			dataRanges = Collections.emptyMap();
		}

		Function<DataField, Feature> function = new Function<DataField, Feature>(){

			@Override
			public ContinuousFeature apply(DataField dataField){
				Number[] range = dataRanges.get(dataField.requireName());

				encoder.addDecorator(dataField, new OutlierDecorator(outlierTreatment, lowValue, highValue));

				if(range != null){
					Interval interval = new Interval(Interval.Closure.CLOSED_CLOSED, range[0], range[1]);

					dataField.addIntervals(interval);
				}

				return new ContinuousFeature(encoder, dataField);
			}
		};

		return super.encodeFeatures(function, encoder);
	}

	static
	private OutlierTreatmentMethod parseOutlierTreatment(String outlierTreatment){
		return OutlierTreatmentMethod.fromValue(outlierTreatment);
	}
}