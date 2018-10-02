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
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.expressions.If;
import org.apache.spark.sql.catalyst.expressions.Multiply;
import org.apache.spark.sql.catalyst.expressions.Or;
import org.apache.spark.sql.catalyst.parser.ParseException;
import org.apache.spark.sql.catalyst.parser.ParserInterface;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.internal.SessionState;
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
		ExpressionTranslator.DataTypeResolver dataTypeResolver = new ExpressionTranslator.DataTypeResolver(){

			@Override
			public DataType getDataType(String name){
				return DataType.DOUBLE;
			}
		};

		ExpressionMapping expressionMapping = translate("SELECT (isnull(x1) and not(isnotnull(x2))) FROM __THIS__", dataTypeResolver);

		checkExpressionMapping(expressionMapping, And.class, Apply.class, DataType.BOOLEAN);

		expressionMapping = translate("SELECT ((x1 <= x2) or (x1 >= x3)) FROM __THIS__", dataTypeResolver);

		checkExpressionMapping(expressionMapping, Or.class, Apply.class, DataType.BOOLEAN);
	}

	@Test
	public void translateArithmeticExpression(){
		ExpressionMapping expressionMapping = translate("SELECT 2 * ((0 + 1) / (0 - 1)) FROM __THIS__", null);

		checkExpressionMapping(expressionMapping, Multiply.class, Apply.class, DataType.INTEGER);
	}

	@Test
	public void translateIfExpression(){
		ExpressionTranslator.DataTypeResolver dataTypeResolver = new ExpressionTranslator.DataTypeResolver(){

			@Override
			public DataType getDataType(String name){

				if(("status").equals(name)){
					return DataType.INTEGER;
				}

				return DataType.DOUBLE;
			}
		};

		ExpressionMapping expressionMapping = translate("SELECT if(status in (1, 2, 3), x1 < 0, x2 > 0)", dataTypeResolver);

		checkExpressionMapping(expressionMapping, If.class, Apply.class, DataType.BOOLEAN);
	}

	static
	private ExpressionMapping translate(String statement, ExpressionTranslator.DataTypeResolver dataTypeResolver){
		SessionState sessionState = ExpressionTranslatorTest.sparkSession.sessionState();

		ParserInterface parserInterface = sessionState.sqlParser();

		LogicalPlan logicalPlan;

		try {
			logicalPlan = parserInterface.parsePlan(statement);
		} catch(ParseException pe){
			throw new IllegalArgumentException(pe);
		}

		List<Expression> expressions = JavaConversions.seqAsJavaList(logicalPlan.expressions());
		if(expressions.size() != 1){
			throw new IllegalArgumentException();
		}

		return ExpressionTranslator.translate(expressions.get(0), dataTypeResolver);
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