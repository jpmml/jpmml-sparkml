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

import java.util.Objects;

import com.google.common.base.Objects.ToStringHelper;
import org.dmg.pmml.TypeDefinitionField;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;

public class DocumentFeature extends Feature {

	private String wordSeparatorRE = null;


	public DocumentFeature(SparkMLEncoder encoder, TypeDefinitionField field, String wordSeparatorRE){
		super(encoder, field.getName(), field.getDataType());

		setWordSeparatorRE(wordSeparatorRE);
	}

	@Override
	public ContinuousFeature toContinuousFeature(){
		throw new UnsupportedOperationException();
	}

	@Override
	public int hashCode(){
		return (31 * super.hashCode()) + Objects.hashCode(this.getWordSeparatorRE());
	}

	@Override
	public boolean equals(Object object){

		if(object instanceof DocumentFeature){
			DocumentFeature that = (DocumentFeature)object;

			return super.equals(object) && Objects.equals(this.getWordSeparatorRE(), that.getWordSeparatorRE());
		}

		return false;
	}

	@Override
	protected ToStringHelper toStringHelper(){
		return super.toStringHelper()
			.add("wordSeparatorRE", getWordSeparatorRE());
	}

	public String getWordSeparatorRE(){
		return this.wordSeparatorRE;
	}

	private void setWordSeparatorRE(String wordSeparatorRE){

		if(wordSeparatorRE == null){
			throw new IllegalArgumentException();
		}

		this.wordSeparatorRE = wordSeparatorRE;
	}
}