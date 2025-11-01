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
import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.InvalidValueTreatmentMethod;
import org.dmg.pmml.MissingValueTreatmentMethod;
import org.dmg.pmml.Value;
import org.jpmml.converter.Feature;
import org.jpmml.converter.FieldUtil;
import org.jpmml.converter.InvalidValueDecorator;
import org.jpmml.converter.MissingValueDecorator;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.SparkMLEncoder;

abstract
public class DomainModelConverter<T extends DomainModel<T>> extends FeatureConverter<T> {

	public DomainModelConverter(T transformer){
		super(transformer);
	}

	protected List<Feature> encodeFeatures(DomainManager domainManager, SparkMLEncoder encoder){
		T transformer = getTransformer();

		Object[] missingValues = transformer.getMissingValues();

		MissingValueTreatmentMethod missingValueTreatment = parseMissingValueTreatment(transformer.getMissingValueTreatment());
		Object missingValueReplacement = transformer.getMissingValueReplacement();
		InvalidValueTreatmentMethod invalidValueTreatment = parseInvalidValueTreatment(transformer.getInvalidValueTreatment());
		Object invalidValueReplacement = transformer.getInvalidValueReplacement();

		List<Feature> result = new ArrayList<>();

		String[] inputCols = transformer.getInputCols();
		for(String inputCol : inputCols){
			Feature feature = encoder.getOnlyFeature(inputCol);

			DataField dataField = domainManager.toDataField(feature);

			FieldUtil.addValues(dataField, Value.Property.MISSING, Arrays.asList(missingValues));

			encoder.addDecorator(dataField, new MissingValueDecorator(missingValueTreatment, missingValueReplacement));
			encoder.addDecorator(dataField, new InvalidValueDecorator(invalidValueTreatment, invalidValueReplacement));

			feature = domainManager.toFeature(dataField);

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
	protected interface DomainManager {

		DataField toDataField(Feature feature);

		Feature toFeature(DataField dataField);
	}
}