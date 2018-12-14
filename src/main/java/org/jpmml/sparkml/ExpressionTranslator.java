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

import java.util.Iterator;
import java.util.List;

import org.apache.spark.sql.catalyst.expressions.Add;
import org.apache.spark.sql.catalyst.expressions.Alias;
import org.apache.spark.sql.catalyst.expressions.And;
import org.apache.spark.sql.catalyst.expressions.AttributeReference;
import org.apache.spark.sql.catalyst.expressions.BinaryArithmetic;
import org.apache.spark.sql.catalyst.expressions.BinaryComparison;
import org.apache.spark.sql.catalyst.expressions.BinaryOperator;
import org.apache.spark.sql.catalyst.expressions.CaseWhen;
import org.apache.spark.sql.catalyst.expressions.Cast;
import org.apache.spark.sql.catalyst.expressions.Divide;
import org.apache.spark.sql.catalyst.expressions.EqualTo;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.expressions.GreaterThan;
import org.apache.spark.sql.catalyst.expressions.GreaterThanOrEqual;
import org.apache.spark.sql.catalyst.expressions.If;
import org.apache.spark.sql.catalyst.expressions.In;
import org.apache.spark.sql.catalyst.expressions.IsNotNull;
import org.apache.spark.sql.catalyst.expressions.IsNull;
import org.apache.spark.sql.catalyst.expressions.LessThan;
import org.apache.spark.sql.catalyst.expressions.LessThanOrEqual;
import org.apache.spark.sql.catalyst.expressions.Literal;
import org.apache.spark.sql.catalyst.expressions.Multiply;
import org.apache.spark.sql.catalyst.expressions.Not;
import org.apache.spark.sql.catalyst.expressions.Or;
import org.apache.spark.sql.catalyst.expressions.Subtract;
import org.apache.spark.sql.catalyst.expressions.UnaryExpression;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.HasDataType;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.visitors.ExpressionCompactor;
import scala.Option;
import scala.Tuple2;
import scala.collection.JavaConversions;

public class ExpressionTranslator {

	static
	public org.dmg.pmml.Expression translate(Expression expression){
		return translate(expression, true);
	}

	static
	public org.dmg.pmml.Expression translate(Expression expression, boolean compact){
		org.dmg.pmml.Expression pmmlExpression = translateInternal(expression);

		if(compact){
			ExpressionCompactor expressionCompactor = new ExpressionCompactor();

			expressionCompactor.applyTo(pmmlExpression);
		}

		return pmmlExpression;
	}

	static
	private org.dmg.pmml.Expression translateInternal(Expression expression){

		if(expression instanceof Alias){
			Alias alias = (Alias)expression;

			Expression child = alias.child();

			return translateInternal(child);
		} // End if

		if(expression instanceof AttributeReference){
			AttributeReference attributeReference = (AttributeReference)expression;

			String name = attributeReference.name();

			return new FieldRef(FieldName.create(name));
		} else

		if(expression instanceof BinaryOperator){
			BinaryOperator binaryOperator = (BinaryOperator)expression;

			String symbol = binaryOperator.symbol();

			Expression left = binaryOperator.left();
			Expression right = binaryOperator.right();

			if(expression instanceof And || expression instanceof Or){

				switch(symbol){
					case "&&":
						symbol = "and";
						break;
					case "||":
						symbol = "or";
						break;
					default:
						throw new IllegalArgumentException(String.valueOf(binaryOperator));
				}
			} else

			if(expression instanceof Add || expression instanceof Divide || expression instanceof Multiply || expression instanceof Subtract){
				BinaryArithmetic binaryArithmetic = (BinaryArithmetic)binaryOperator;

				switch(symbol){
					case "+":
					case "/":
					case "*":
					case "-":
						break;
					default:
						throw new IllegalArgumentException(String.valueOf(binaryArithmetic));
				}
			} else

			if(expression instanceof EqualTo || expression instanceof GreaterThan || expression instanceof GreaterThanOrEqual || expression instanceof LessThan || expression instanceof LessThanOrEqual){
				BinaryComparison binaryComparison = (BinaryComparison)binaryOperator;

				switch(symbol){
					case "=":
						symbol = "equal";
						break;
					case ">":
						symbol = "greaterThan";
						break;
					case ">=":
						symbol = "greaterOrEqual";
						break;
					case "<":
						symbol = "lessThan";
						break;
					case "<=":
						symbol = "lessOrEqual";
						break;
					default:
						throw new IllegalArgumentException(String.valueOf(binaryComparison));
				}
			} else

			{
				throw new IllegalArgumentException(String.valueOf(binaryOperator));
			}

			return PMMLUtil.createApply(symbol, translateInternal(left), translateInternal(right));
		} else

		if(expression instanceof CaseWhen){
			CaseWhen caseWhen = (CaseWhen)expression;

			List<Tuple2<Expression, Expression>> branches = JavaConversions.seqAsJavaList(caseWhen.branches());

			Option<Expression> elseValue = caseWhen.elseValue();

			Apply apply = null;

			Iterator<Tuple2<Expression, Expression>> branchIt = branches.iterator();

			Apply prevBranchApply = null;

			do {
				Tuple2<Expression, Expression> branch = branchIt.next();

				Expression predicate = branch._1();
				Expression value = branch._2();

				Apply branchApply = PMMLUtil.createApply("if")
					.addExpressions(translateInternal(predicate), translateInternal(value));

				if(apply == null){
					apply = branchApply;
				} // End if

				if(prevBranchApply != null){
					prevBranchApply.addExpressions(branchApply);
				}

				prevBranchApply = branchApply;
			} while(branchIt.hasNext());

			if(elseValue.isDefined()){
				Expression value = elseValue.get();

				prevBranchApply.addExpressions(translateInternal(value));
			}

			return apply;
		} else

		if(expression instanceof Cast){
			Cast cast = (Cast)expression;

			Expression child = cast.child();

			DataType dataType = DatasetUtil.translateDataType(cast.dataType());

			org.dmg.pmml.Expression pmmlExpression = translateInternal(child);

			if(pmmlExpression instanceof HasDataType){
				HasDataType<?> hasDataType = (HasDataType<?>)pmmlExpression;

				hasDataType.setDataType(dataType);

				return pmmlExpression;
			} else

			{
				throw new IllegalArgumentException(String.valueOf(cast));
			}
		} else

		if(expression instanceof If){
			If _if = (If)expression;

			Expression predicate = _if.predicate();

			Expression trueValue = _if.trueValue();
			Expression falseValue = _if.falseValue();

			return PMMLUtil.createApply("if", translateInternal(predicate))
				.addExpressions(translateInternal(trueValue), translateInternal(falseValue));
		} else

		if(expression instanceof In){
			In in = (In)expression;

			Expression value = in.value();

			List<Expression> elements = JavaConversions.seqAsJavaList(in.list());

			Apply apply = PMMLUtil.createApply("isIn", translateInternal(value));

			for(Expression element : elements){
				apply.addExpressions(translateInternal(element));
			}

			return apply;
		} else

		if(expression instanceof Literal){
			Literal literal = (Literal)expression;

			Object value = literal.value();

			DataType dataType = DatasetUtil.translateDataType(literal.dataType());

			return PMMLUtil.createConstant(value, dataType);
		} else

		if(expression instanceof Not){
			 Not not = (Not)expression;

			 Expression child = not.child();

			 return PMMLUtil.createApply("not", translateInternal(child));
		} else

		if(expression instanceof UnaryExpression){
			UnaryExpression unaryExpression = (UnaryExpression)expression;

			Expression child = unaryExpression.child();

			if(expression instanceof IsNotNull){
				return PMMLUtil.createApply("isNotMissing", translateInternal(child));
			} else

			if(expression instanceof IsNull){
				return PMMLUtil.createApply("isMissing", translateInternal(child));
			} else

			{
				throw new IllegalArgumentException(String.valueOf(unaryExpression));
			}
		} else

		{
			throw new IllegalArgumentException(String.valueOf(expression));
		}
	}
}