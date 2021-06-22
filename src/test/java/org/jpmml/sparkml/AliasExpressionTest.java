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

import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.model.ReflectionUtil;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class AliasExpressionTest {

	@Test
	public void unwrap(){
		FieldRef fieldRef = new FieldRef(FieldName.create("x"));

		Expression expression = new AliasExpression("parent", new AliasExpression("child", fieldRef));

		checkExpression(fieldRef, expression);

		expression = new AliasExpression("parent", PMMLUtil.createApply(PMMLFunctions.ADD, new AliasExpression("left child", fieldRef), new AliasExpression("right child", fieldRef)));

		checkExpression(PMMLUtil.createApply(PMMLFunctions.ADD, fieldRef, fieldRef), expression);
	}

	static
	private void checkExpression(Expression expected, Expression expression){
		expression = AliasExpression.unwrap(expression);

		assertTrue(ReflectionUtil.equals(expected, expression));
	}
}