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
package org.jpmml.sparkml;

public class TermUtil {

	private TermUtil(){
	}

	static
	public boolean hasPunctuation(String string){
		String[] tokens = string.split("\\s+");

		for(String token : tokens){
			int length = token.length();

			if(length > 0){
				char first = token.charAt(0);
				char last = token.charAt(length - 1);

				if(isPunctuation(first) || isPunctuation(last)){
					return true;
				}
			}
		}

		return false;
	}

	static
	public boolean isPunctuation(char c){
		int type = Character.getType(c);

		switch(type){
			case Character.DASH_PUNCTUATION:
			case Character.END_PUNCTUATION:
			case Character.START_PUNCTUATION:
			case Character.CONNECTOR_PUNCTUATION:
			case Character.OTHER_PUNCTUATION:
			case Character.INITIAL_QUOTE_PUNCTUATION:
			case Character.FINAL_QUOTE_PUNCTUATION:
				return true;
			default:
				return false;
		}
	}
}