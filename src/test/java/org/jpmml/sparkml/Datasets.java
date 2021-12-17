/*
 * Copyright (c) 2021 Villu Ruusmann
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

import org.jpmml.converter.FieldNameUtil;

interface Datasets {

	String AUDIT = "Audit";
	String AUTO = "Auto";
	String HOUSING = "Housing";
	String IRIS = "Iris";
	String SENTIMENT = "Sentiment";
	String SHOPPING = "Shopping";
	String VISIT = "Visit";

	String AUDIT_PROBABILITY_TRUE = FieldNameUtil.create("probability", 1);
	String AUDIT_PROBABILITY_FALSE = FieldNameUtil.create("probability", 0);

	String IRIS_PROBABILITY_SETOSA = FieldNameUtil.create("probability", "setosa");
	String IRIS_PROBABILITY_VERSICOLOR = FieldNameUtil.create("probability", "versicolor");
	String IRIS_PROBABILITY_VIRGINICA = FieldNameUtil.create("probability", "virginica");
}