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
package org.jpmml.sparkml;

import java.util.regex.Pattern;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class RegexKeyTest {

	@Test
	public void compile(){
		RegexKey anyKey = new RegexKey(Pattern.compile(".*"));
		RegexKey dotAsteriskKey = new RegexKey(Pattern.compile(".*", Pattern.LITERAL));

		assertTrue(anyKey.test(""));
		assertTrue(anyKey.test(".*"));

		assertFalse(dotAsteriskKey.test(""));
		assertTrue(dotAsteriskKey.test(".*"));
	}
}