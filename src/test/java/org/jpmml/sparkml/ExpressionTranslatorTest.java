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

import java.util.List;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.And;
import org.apache.spark.sql.catalyst.expressions.Divide;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.expressions.If;
import org.apache.spark.sql.catalyst.expressions.Or;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.collection.JavaConversions;

import static org.junit.Assert.assertEquals;

public class ExpressionTranslatorTest {

	@Test
	public void translateLogicalExpression(){
		ExpressionMapping expressionMapping = translate("SELECT (isnull(x1) and not(isnotnull(x2))) FROM __THIS__");

		checkExpressionMapping(expressionMapping, And.class, Apply.class, DataType.BOOLEAN);

		expressionMapping = translate("SELECT ((x1 <= x3) or (x2 >= x3)) FROM __THIS__");

		checkExpressionMapping(expressionMapping, Or.class, Apply.class, DataType.BOOLEAN);
	}

	@Test
	public void translateArithmeticExpression(){
		ExpressionMapping expressionMapping = translate("SELECT ((x1 - x3) / (x2 + x3)) FROM __THIS__");

		checkExpressionMapping(expressionMapping, Divide.class, Apply.class, DataType.DOUBLE);
	}

	@Test
	public void translateIfExpression(){
		ExpressionMapping expressionMapping = translate("SELECT if(flag, x1 != x3, x2 != x3) FROM __THIS__");

		checkExpressionMapping(expressionMapping, If.class, Apply.class, DataType.BOOLEAN);
	}

	static
	private ExpressionMapping translate(String statement){
		StructType schema = new StructType()
			.add("flag", DataTypes.BooleanType)
			.add("x1", DataTypes.DoubleType)
			.add("x2", DataTypes.DoubleType)
			.add("x3", DataTypes.DoubleType);

		LogicalPlan logicalPlan = DatasetUtil.createAnalyzedLogicalPlan(ExpressionTranslatorTest.sparkSession, schema, statement);

		List<Expression> expressions = JavaConversions.seqAsJavaList(logicalPlan.expressions());
		if(expressions.size() != 1){
			throw new IllegalArgumentException();
		}

		return ExpressionTranslator.translate(expressions.get(0));
	}

	static
	private void checkExpressionMapping(ExpressionMapping expressionMapping, Class<? extends Expression> fromClazz, Class<? extends org.dmg.pmml.Expression> toClazz, DataType dataType){
		Expression from = expressionMapping.getFrom();
		org.dmg.pmml.Expression to = expressionMapping.getTo();

		assertEquals(fromClazz, from.getClass());
		assertEquals(toClazz, to.getClass());
		assertEquals(dataType, expressionMapping.getDataType());
	}

	@BeforeClass
	static
	public void createSparkSession(){
		ExpressionTranslatorTest.sparkSession = SparkSessionUtil.createSparkSession();
	}

	@AfterClass
	static
	public void destroySparkSession(){
		ExpressionTranslatorTest.sparkSession = SparkSessionUtil.destroySparkSession(ExpressionTranslatorTest.sparkSession);
	}

	public static SparkSession sparkSession = null;
}