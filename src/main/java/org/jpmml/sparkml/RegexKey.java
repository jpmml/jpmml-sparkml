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

import java.util.Objects;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RegexKey implements Predicate<String> {

	private Pattern pattern = null;


	public RegexKey(Pattern pattern){
		setPattern(pattern);
	}

	@Override
	public boolean test(String string){
		Pattern pattern = getPattern();

		Matcher matcher = pattern.matcher(string);

		return matcher.matches();
	}

	@Override
	public int hashCode(){
		return hashCode(getPattern());
	}

	@Override
	public boolean equals(Object object){

		if(object instanceof RegexKey){
			RegexKey that = (RegexKey)object;

			return equals(this.getPattern(), that.getPattern());
		}

		return false;
	}

	public Pattern getPattern(){
		return this.pattern;
	}

	private void setPattern(Pattern pattern){
		this.pattern = pattern;
	}

	static
	private int hashCode(Pattern pattern){
		return Objects.hash(pattern.pattern(), pattern.flags());
	}

	static
	private boolean equals(Pattern left, Pattern right){
		return Objects.equals(left.pattern(), right.pattern()) && Objects.equals(left.flags(), right.flags());
	}
}