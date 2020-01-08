/*
 * Copyright (c) 2018 Villu Ruusmann
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
package org.jpmml.sparkml.model;

import java.util.LinkedHashMap;
import java.util.Map;

import org.jpmml.converter.HasNativeConfiguration;
import org.jpmml.sparkml.HasSparkMLOptions;
import org.jpmml.sparkml.visitors.TreeModelCompactor;

public interface HasTreeOptions extends HasSparkMLOptions, HasNativeConfiguration {

	/**
	 * @see TreeModelCompactor
	 */
	String OPTION_COMPACT = "compact";

	@Override
	default
	public Map<String, ?> getNativeConfiguration(){
		Map<String, Object> result = new LinkedHashMap<>();
		result.put(HasTreeOptions.OPTION_COMPACT, Boolean.FALSE);

		return result;
	}
}