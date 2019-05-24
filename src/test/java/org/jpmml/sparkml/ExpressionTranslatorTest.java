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

import java.util.Collections;
import java.util.List;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.FieldRef;
import org.jpmml.evaluator.EvaluationContext;
import org.jpmml.evaluator.ExpressionUtil;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.FieldValueUtil;
import org.jpmml.evaluator.VirtualEvaluationContext;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.collection.JavaConversions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class ExpressionTranslatorTest {

	@Test
	public void translateLogicalExpression(){
		Apply apply = (Apply)translate("isnull(x1) and not(isnotnull(x2))");

		checkApply(apply, "and", Apply.class, Apply.class);

		apply = (Apply)translate("(x1 <= 0) or (x2 >= 0)");

		checkApply(apply, "or", Apply.class, Apply.class);
	}

	@Test
	public void evaluateArithmeticExpression(){
		checkValue(3, "1 + int(2)");
		checkValue(3d, "cast(1.0 as double) + double(2.0)");

		checkValue(1, "2 - int(1)");
		checkValue(1d, "cast(2.0 as double) - double(1.0)");

		checkValue(6, "2 * int(3)");
		checkValue(6d, "cast(2.0 as double) * double(3.0)");

		// "Always perform floating point division"
		checkValue(1.5d, "3 / int(2)");
		checkValue(1d, "2 / int(2)");
	}

	@Test
	public void translateArithmeticExpression(){
		Apply apply = (Apply)translate("-((x1 - 1) / (x2 + 1))");

		List<org.dmg.pmml.Expression> pmmlExpressions = checkApply(apply, "*", Constant.class, Apply.class);

		apply = (Apply)pmmlExpressions.get(1);

		checkApply(apply, "/", Apply.class, Apply.class);
	}

	@Test
	public void translateCaseWhenExpression(){
		Apply apply = (Apply)translate("CASE WHEN x1 < 0 THEN x1 WHEN x2 > 0 THEN x2 ELSE 0 END");

		List<org.dmg.pmml.Expression> pmmlExpressions = checkApply(apply, "if", Apply.class, FieldRef.class, Apply.class);

		apply = (Apply)pmmlExpressions.get(0);

		checkApply(apply, "lessThan", FieldRef.class, Constant.class);

		apply = (Apply)pmmlExpressions.get(2);

		checkApply(apply, "if", Apply.class, FieldRef.class, Constant.class);
	}

	@Test
	public void translateIfExpression(){
		Apply apply = (Apply)translate("if(status in (-1, 1), x1 != 0, x2 != 0)");

		checkApply(apply, "if", Apply.class, Apply.class, Apply.class);
	}

	@Test
	public void evaluateUnaryMathExpression(){
		checkValue(1, "-int(-1)");
		checkValue(1d, "-double(-1.0)");

		checkValue(1, "abs(-1)");

		checkValue(0d, "ceil(double(-0.1))");
		checkValue(5d, "ceil(5)");

		checkValue(1.0d, "exp(0)");

		checkValue(-1d, "floor(double(-0.1))");
		checkValue(5d, "floor(5)");

		checkValue(0.0d, "ln(1)");

		checkValue(1.0d, "log10(10)");

		checkValue(1, "negative(-1)");
		checkValue(-1, "negative(1)");

		checkValue(-1, "positive(-1)");
		checkValue(1, "positive(1)");

		checkValue(8.0d, "pow(2, 3)");

		checkValue(12d, "rint(double(12.3456))");

		checkValue(2.0d, "sqrt(4)");
	}

	@Test
	public void evaluateStringExpression(){
		checkValue("SparkSQL", "concat(\"Spark\", \"SQL\")");

		checkValue("sparksql", "lower(\"SparkSql\")");

		checkValue("k SQL", "substr(\"Spark SQL\", 5)");

		try {
			checkValue("SQL", "substr(\"Spark SQL\", -3)");

			fail();
		} catch(IllegalArgumentException iae){
			// Ignored
		}

		checkValue("k", "substr(\"Spark SQL\", 5, 1)");

		checkValue("SparkSQL", "trim(\"    SparkSQL   \")");

		checkValue("SPARKSQL", "upper(\"SparkSql\")");
	}

	static
	private Object evaluate(String sqlExpression){
		Expression expression = translateInternal("SELECT (" + sqlExpression + ") FROM __THIS__");

		return expression.eval(InternalRow.empty());
	}

	static
	private org.dmg.pmml.Expression translate(String sqlExpression){
		Expression expression = translateInternal("SELECT (" + sqlExpression + ") FROM __THIS__");

		return ExpressionTranslator.translate(expression);
	}

	static
	private Expression translateInternal(String sqlStatement){
		StructType schema = new StructType()
			.add("flag", DataTypes.BooleanType)
			.add("x1", DataTypes.DoubleType)
			.add("x2", DataTypes.DoubleType)
			.add("status", DataTypes.IntegerType);

		LogicalPlan logicalPlan = DatasetUtil.createAnalyzedLogicalPlan(ExpressionTranslatorTest.sparkSession, schema, sqlStatement);

		List<Expression> expressions = JavaConversions.seqAsJavaList(logicalPlan.expressions());
		if(expressions.size() != 1){
			throw new IllegalArgumentException();
		}

		return expressions.get(0);
	}

	static
	public void checkValue(Object expectedValue, String sqlExpression){
		Expression expression = translateInternal("SELECT (" + sqlExpression + ") FROM __THIS__");

		Object sparkValue = expression.eval(InternalRow.empty());

		if(expectedValue instanceof String){
			assertEquals(expectedValue, sparkValue.toString());
		} else

		if(expectedValue instanceof Integer){
			assertEquals(expectedValue, ((Number)sparkValue).intValue());
		} else

		if(expectedValue instanceof Float){
			assertEquals(expectedValue, ((Number)sparkValue).floatValue());
		} else

		if(expectedValue instanceof Double){
			assertEquals(expectedValue, ((Number)sparkValue).doubleValue());
		} else

		{
			assertEquals(expectedValue, sparkValue);
		}

		org.dmg.pmml.Expression pmmlExpression = ExpressionTranslator.translate(expression);

		EvaluationContext context = new VirtualEvaluationContext();
		context.declareAll(Collections.emptyMap());

		FieldValue value = ExpressionUtil.evaluate(pmmlExpression, context);

		Object pmmlValue = FieldValueUtil.getValue(value);
		assertEquals(expectedValue, pmmlValue);
	}

	static
	private List<org.dmg.pmml.Expression> checkApply(Apply apply, String function, Class<? extends org.dmg.pmml.Expression>... pmmlExpressionClazzes){
		assertEquals(function, apply.getFunction());

		List<org.dmg.pmml.Expression> pmmlExpressions = apply.getExpressions();
		assertEquals(pmmlExpressionClazzes.length, pmmlExpressions.size());

		for(int i = 0; i < pmmlExpressionClazzes.length; i++){
			Class<? extends org.dmg.pmml.Expression> expressionClazz = pmmlExpressionClazzes[i];
			org.dmg.pmml.Expression pmmlExpression = pmmlExpressions.get(i);

			assertEquals(expressionClazz, pmmlExpression.getClass());
		}

		return pmmlExpressions;
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