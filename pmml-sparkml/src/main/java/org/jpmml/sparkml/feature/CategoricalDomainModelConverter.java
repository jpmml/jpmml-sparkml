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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.dmg.pmml.DataField;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldUtil;
import org.jpmml.converter.ObjectFeature;
import org.jpmml.sparkml.SparkMLEncoder;

public class CategoricalDomainModelConverter extends DomainModelConverter<CategoricalDomainModel> {

	public CategoricalDomainModelConverter(CategoricalDomainModel transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(SparkMLEncoder encoder){
		CategoricalDomainModel transformer = getTransformer();

		boolean withData = transformer.getWithData();

		Map<String, Object[]> dataValues;

		if(withData){
			dataValues = DomainUtil.toJavaMap(transformer.getDataValues());
		} else

		{
			dataValues = Collections.emptyMap();
		}

		MissingValueTreatmentMethod missingValueTreatment = parseMissingValueTreatment(transformer.getMissingValueTreatment());
		InvalidValueTreatmentMethod invalidValueTreatment = parseInvalidValueTreatment(transformer.getInvalidValueTreatment());

		Set<Object> replacementValues = new HashSet<>();

		switch(missingValueTreatment){
			case AS_MEAN:
			case AS_MODE:
			case AS_MEDIAN:
			case AS_VALUE:
				{
					Object missingValueReplacement = transformer.getMissingValueReplacement();

					replacementValues.add(missingValueReplacement);
				}
				break;
			default:
				break;
		} // End switch

		switch(invalidValueTreatment){
			case AS_VALUE:
				{
					Object invalidValueReplacement = transformer.getInvalidValueReplacement();

					replacementValues.add(invalidValueReplacement);
				}
				break;
			default:
				break;
		}

		DomainManager domainManager = new DomainManager(){

			@Override
			public DataField toDataField(Feature feature){
				Object[] values = dataValues.get(feature.getName());

				DataField dataField = (DataField)encoder.toCategorical(feature, values != null ? Arrays.asList(values) : null);

				List<Object> extraValues = new ArrayList<>(replacementValues);
				extraValues.removeAll(Arrays.asList(values));

				if(!extraValues.isEmpty()){
					FieldUtil.addValues(dataField, extraValues);
				}

				return dataField;
			}

			@Override
			public ObjectFeature toFeature(DataField dataField){
				return new ObjectFeature(encoder, dataField);
			}
		};

		return super.encodeFeatures(domainManager, encoder);
	}
}