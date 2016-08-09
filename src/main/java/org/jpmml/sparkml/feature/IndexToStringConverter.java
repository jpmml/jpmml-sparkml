/*
 * Copyright (c) 2016 Villu Ruusmann
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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.feature.IndexToString;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Value;
import org.jpmml.converter.Feature;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.WildcardFeature;
import org.jpmml.sparkml.FeatureConverter;
import org.jpmml.sparkml.FeatureMapper;

public class IndexToStringConverter extends FeatureConverter<IndexToString> {

	public IndexToStringConverter(IndexToString transformer){
		super(transformer);
	}

	@Override
	public List<Feature> encodeFeatures(FeatureMapper featureMapper){
		IndexToString transformer = getTransformer();

		DataField dataField = featureMapper.createDataField(formatName(transformer), OpType.CATEGORICAL, DataType.STRING);

		String[] labels = transformer.getLabels();
		if(labels != null && labels.length > 0){
			List<Value> values = dataField.getValues();

			values.addAll(PMMLUtil.createValues(Arrays.asList(labels)));
		}

		Feature feature = new WildcardFeature(dataField);

		return Collections.singletonList(feature);
	}
}