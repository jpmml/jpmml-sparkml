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
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.FieldRef;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.collection.JavaConversions;

import static org.junit.Assert.assertEquals;

public class ExpressionTranslatorTest {

	@Test
	public void translateLogicalExpression(){
		Apply apply = (Apply)translate("SELECT (isnull(x1) and not(isnotnull(x2))) FROM __THIS__");

		checkApply(apply, "and", Apply.class, Apply.class);

		apply = (Apply)translate("SELECT ((x1 <= 0) or (x2 >= 0)) FROM __THIS__");

		checkApply(apply, "or", Apply.class, Apply.class);
	}

	@Test
	public void translateArithmeticExpression(){
		Apply apply = (Apply)translate("SELECT ((x1 - 1) / (x2 + 1)) FROM __THIS__");

		checkApply(apply, "/", Apply.class, Apply.class);
	}

	@Test
	public void translateCaseWhenExpression(){
		Apply apply = (Apply)translate("SELECT (CASE WHEN x1 < 0 THEN x1 WHEN x2 > 0 THEN x2 ELSE 0 END) FROM __THIS__");

		List<org.dmg.pmml.Expression> pmmlExpressions = checkApply(apply, "if", Apply.class, FieldRef.class, Apply.class);

		apply = (Apply)pmmlExpressions.get(0);

		checkApply(apply, "lessThan", FieldRef.class, Constant.class);

		apply = (Apply)pmmlExpressions.get(2);

		checkApply(apply, "if", Apply.class, FieldRef.class, Constant.class);
	}

	@Test
	public void translateIfExpression(){
		Apply apply = (Apply)translate("SELECT if(status in (0, 1), x1 != 0, x2 != 0) FROM __THIS__");

		checkApply(apply, "if", Apply.class, Apply.class, Apply.class);
	}

	static
	private org.dmg.pmml.Expression translate(String statement){
		StructType schema = new StructType()
			.add("flag", DataTypes.BooleanType)
			.add("x1", DataTypes.DoubleType)
			.add("x2", DataTypes.DoubleType)
			.add("status", DataTypes.IntegerType);

		LogicalPlan logicalPlan = DatasetUtil.createAnalyzedLogicalPlan(ExpressionTranslatorTest.sparkSession, schema, statement);

		List<Expression> expressions = JavaConversions.seqAsJavaList(logicalPlan.expressions());
		if(expressions.size() != 1){
			throw new IllegalArgumentException();
		}

		return ExpressionTranslator.translate(expressions.get(0));
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