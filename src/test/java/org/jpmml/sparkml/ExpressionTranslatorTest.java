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
import java.util.Objects;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.PMML;
import org.dmg.pmml.PMMLFunctions;
import org.dmg.pmml.TransformationDictionary;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.evaluator.EvaluationContext;
import org.jpmml.evaluator.ExpressionUtil;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.FieldValueUtil;
import org.jpmml.evaluator.VirtualEvaluationContext;
import org.jpmml.model.ReflectionUtil;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.collection.JavaConversions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class ExpressionTranslatorTest {

	@Test
	public void translateLogicalExpression(){
		String string = "isnull(x1) and not(isnotnull(x2))";

		FieldRef first = new FieldRef(FieldName.create("x1"));
		FieldRef second = new FieldRef(FieldName.create("x2"));

		Apply expected = PMMLUtil.createApply(PMMLFunctions.AND,
			PMMLUtil.createApply(PMMLFunctions.ISMISSING, first),
			// "not(isnotnull(..)) -> "isnull(..)"
			PMMLUtil.createApply(PMMLFunctions.ISMISSING, second)
		);

		checkExpression(expected, string);

		string = "(x1 <= 0) or (x2 >= 0)";

		expected = PMMLUtil.createApply(PMMLFunctions.OR,
			PMMLUtil.createApply(PMMLFunctions.LESSOREQUAL, first, PMMLUtil.createConstant(0, DataType.DOUBLE)),
			PMMLUtil.createApply(PMMLFunctions.GREATEROREQUAL, second, PMMLUtil.createConstant(0, DataType.DOUBLE))
		);

		checkExpression(expected, string);
	}

	@Test
	public void evaluateArithmeticExpression(){
		checkValue(3, "1 + int(2)");
		checkValue("3", "cast(1 + int(2) as string) as int_string");
		checkValue(3d, "cast(1.0 as double) + double(2.0)");

		checkValue(1, "2 - int(1)");
		checkValue("1", "cast(2 - int(1) as string) as int_string");
		checkValue(1d, "cast(2.0 as double) - double(1.0)");

		checkValue(6, "2 * int(3)");
		checkValue("6", "cast(2 * int(3) as string) as int_string");
		checkValue(6d, "cast(2.0 as double) * double(3.0)");

		// "Always perform floating point division"
		checkValue(1.5d, "3 / int(2)");
		checkValue(1d, "2 / int(2)");
	}

	@Test
	public void translateArithmeticExpression(){
		String string = "-((x1 - 1) / (x2 + 1))";

		Apply expected = PMMLUtil.createApply(PMMLFunctions.MULTIPLY,
			PMMLUtil.createConstant(-1),
			PMMLUtil.createApply(PMMLFunctions.DIVIDE,
				PMMLUtil.createApply(PMMLFunctions.SUBTRACT, new FieldRef(FieldName.create("x1")), PMMLUtil.createConstant(1, DataType.DOUBLE)),
				PMMLUtil.createApply(PMMLFunctions.ADD, new FieldRef(FieldName.create("x2")), PMMLUtil.createConstant(1, DataType.DOUBLE))
			)
		);

		checkExpression(expected, string);
	}

	@Test
	public void translateCaseWhenExpression(){
		String string = "CASE WHEN x1 < 0 THEN x1 WHEN x2 > 0 THEN x2 ELSE 0 END";

		FieldRef first = new FieldRef(FieldName.create("x1"));
		FieldRef second = new FieldRef(FieldName.create("x2"));

		Constant zero = PMMLUtil.createConstant(0, DataType.DOUBLE);

		Apply expected = PMMLUtil.createApply(PMMLFunctions.IF,
			PMMLUtil.createApply(PMMLFunctions.LESSTHAN, first, zero),
			first,
			PMMLUtil.createApply(PMMLFunctions.IF,
				PMMLUtil.createApply(PMMLFunctions.GREATERTHAN, second, zero),
				second,
				zero
			)
		);

		checkExpression(expected, string);
	}

	@Test
	public void translateIfExpression(){
		String string = "if(status in (-1, 1), x1 != 0, x2 != 0)";

		Apply expected = PMMLUtil.createApply(PMMLFunctions.IF,
			PMMLUtil.createApply(PMMLFunctions.ISIN, new FieldRef(FieldName.create("status")), PMMLUtil.createConstant(-1), PMMLUtil.createConstant(1)),
			PMMLUtil.createApply(PMMLFunctions.NOTEQUAL, new FieldRef(FieldName.create("x1")), PMMLUtil.createConstant(0, DataType.DOUBLE)),
			PMMLUtil.createApply(PMMLFunctions.NOTEQUAL, new FieldRef(FieldName.create("x2")), PMMLUtil.createConstant(0, DataType.DOUBLE))
		);

		checkExpression(expected, string);
	}

	@Test
	public void evaluateUnaryMathExpression(){
		checkValue(1, "-int(-1)");
		checkValue(1d, "-double(-1.0)");

		checkValue(1, "abs(-1)");

		checkValue(0, "ceil(double(-0.1))");
		checkValue(5, "ceil(5)");

		checkValue(1.0d, "exp(0)");

		checkValue(0.0d, "expm1(0)");

		checkValue(-1, "floor(double(-0.1))");
		checkValue(5, "floor(5)");

		checkValue(5.0d, "hypot(3, 4)");

		checkValue(0.0d, "ln(1)");

		checkValue(1.0d, "log10(10)");

		checkValue(0.0d, "log1p(0)");

		checkValue(1, "negative(-1)");
		checkValue(-1, "negative(1)");

		checkValue(-1, "positive(-1)");
		checkValue(1, "positive(1)");

		checkValue(8.0d, "pow(2, 3)");

		checkValue(12d, "rint(double(12.3456))");

		checkValue(2.0d, "sqrt(4)");
	}

	@Test
	public void evaluateTrigonometricExpression(){
		checkValue(1.0d, "cos(0)");
		checkValue(0.0d, "acos(1.0)");
		checkValue(1.0d, "cosh(0)");

		checkValue(0.0d, "sin(0)");
		checkValue(0.0d, "asin(0.0)");
		checkValue(0.0d, "sinh(0)");

		checkValue(0.0d, "tan(0)");
		checkValue(0.0d, "atan(0.0)");
		checkValue(0.0d, "tanh(0)");
	}

	@Test
	public void translateAggregationExpression(){
		checkValue(10, "greatest(10, 9, 2, 4, 3)");

		checkValue(2, "least(10, 9, 2, 4, 3)");
	}

	@Test
	public void evaluateRegexExpression(){
		checkValue(true, "\"\\abc\" rlike \"^\\abc$\"");

		checkValue(true, "\"January\" rlike \"ar?y\"");
		checkValue(true, "\"February\" rlike \"ar?y\"");
		checkValue(false, "\"March\" rlike \"ar?y\"");
		checkValue(false, "\"April\" rlike \"ar?y\"");
		checkValue(true, "\"May\" rlike \"ar?y\"");
		checkValue(false, "\"June\" rlike \"ar?y\"");

		checkValue("num-num", "regexp_replace(\"100-200\", \"(\\\\d+)\", \"num\")");

		checkValue("c", "regexp_replace(\"BBBB\", \"B+\", \"c\")");
	}

	@Test
	public void evaluateStringExpression(){
		checkValue(10, "char_length(\"Spark SQL \")");
		checkValue(10, "character_length(\"Spark SQL \")");

		checkValue("SparkSQL", "concat(\"Spark\", \"SQL\")");

		checkValue(10, "length(\"Spark SQL \")");

		checkValue("sparksql", "lcase(\"SparkSql\")");
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

		checkValue("SPARKSQL", "ucase(\"SparkSql\")");
		checkValue("SPARKSQL", "upper(\"SparkSql\")");
	}

	@Test
	public void evaluateValueExpression(){
		checkValue(true, "isnan(cast(\"NaN\" as double))");

		checkValue(true, "isnull(NULL)");
		checkValue(false, "isnull(0)");

		checkValue(false, "isnotnull(NULL)");
		checkValue(true, "isnotnull(0)");
	}

	static
	private org.dmg.pmml.Expression translate(String sqlExpression){
		ConverterFactory converterFactory = new ConverterFactory(Collections.emptyMap());

		SparkMLEncoder encoder = new SparkMLEncoder(ExpressionTranslatorTest.schema, converterFactory);

		Expression expression = translateInternal("SELECT " + sqlExpression + " FROM __THIS__");

		org.dmg.pmml.Expression pmmlExpression = ExpressionTranslator.translate(encoder, expression);

		pmmlExpression = AliasExpression.unwrap(pmmlExpression);

		return pmmlExpression;
	}

	static
	private Object evaluate(String sqlExpression){
		Expression expression = translateInternal("SELECT " + sqlExpression + " FROM __THIS__");

		return expression.eval(InternalRow.empty());
	}

	static
	private Expression translateInternal(String sqlStatement){
		LogicalPlan logicalPlan = DatasetUtil.createAnalyzedLogicalPlan(ExpressionTranslatorTest.sparkSession, ExpressionTranslatorTest.schema, sqlStatement);

		List<Expression> expressions = JavaConversions.seqAsJavaList(logicalPlan.expressions());
		if(expressions.size() != 1){
			throw new IllegalArgumentException();
		}

		return expressions.get(0);
	}

	static
	public void checkValue(Object expectedValue, String sqlExpression){
		ConverterFactory converterFactory = new ConverterFactory(Collections.emptyMap());

		SparkMLEncoder encoder = new SparkMLEncoder(ExpressionTranslatorTest.schema, converterFactory);

		Expression expression = translateInternal("SELECT " + sqlExpression + " FROM __THIS__");

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

		org.dmg.pmml.Expression pmmlExpression = ExpressionTranslator.translate(encoder, expression);

		pmmlExpression = AliasExpression.unwrap(pmmlExpression);

		PMML pmml = encoder.encodePMML();

		EvaluationContext context = new VirtualEvaluationContext(){

			@Override
			public FieldValue resolve(FieldName name){
				TransformationDictionary transformationDictionary = pmml.getTransformationDictionary();

				if(transformationDictionary != null && transformationDictionary.hasDerivedFields()){
					List<DerivedField> derivedFields = transformationDictionary.getDerivedFields();

					for(DerivedField derivedField : derivedFields){

						if(Objects.equals(derivedField.getName(), name)){
							return ExpressionUtil.evaluate(derivedField, this);
						}
					}
				}

				return super.resolve(name);
			}
		};
		context.declareAll(Collections.emptyMap());

		FieldValue value = ExpressionUtil.evaluate(pmmlExpression, context);

		Object pmmlValue = FieldValueUtil.getValue(value);
		assertEquals(expectedValue, pmmlValue);
	}

	static
	private void checkExpression(org.dmg.pmml.Expression expected, String string){
		org.dmg.pmml.Expression actual = translate(string);

		assertTrue(ReflectionUtil.equals(expected, actual));
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

	private static final StructType schema = new StructType()
		.add("flag", DataTypes.BooleanType)
		.add("x1", DataTypes.DoubleType)
		.add("x2", DataTypes.DoubleType)
		.add("status", DataTypes.IntegerType);
}