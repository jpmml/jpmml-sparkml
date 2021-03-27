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

import org.apache.spark.sql.catalyst.expressions.Abs;
import org.apache.spark.sql.catalyst.expressions.Add;
import org.apache.spark.sql.catalyst.expressions.Alias;
import org.apache.spark.sql.catalyst.expressions.And;
import org.apache.spark.sql.catalyst.expressions.AttributeReference;
import org.apache.spark.sql.catalyst.expressions.BinaryArithmetic;
import org.apache.spark.sql.catalyst.expressions.BinaryComparison;
import org.apache.spark.sql.catalyst.expressions.BinaryOperator;
import org.apache.spark.sql.catalyst.expressions.CaseWhen;
import org.apache.spark.sql.catalyst.expressions.Cast;
import org.apache.spark.sql.catalyst.expressions.Ceil;
import org.apache.spark.sql.catalyst.expressions.Concat;
import org.apache.spark.sql.catalyst.expressions.Divide;
import org.apache.spark.sql.catalyst.expressions.EqualTo;
import org.apache.spark.sql.catalyst.expressions.Exp;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.expressions.Floor;
import org.apache.spark.sql.catalyst.expressions.GreaterThan;
import org.apache.spark.sql.catalyst.expressions.GreaterThanOrEqual;
import org.apache.spark.sql.catalyst.expressions.If;
import org.apache.spark.sql.catalyst.expressions.In;
import org.apache.spark.sql.catalyst.expressions.IsNotNull;
import org.apache.spark.sql.catalyst.expressions.IsNull;
import org.apache.spark.sql.catalyst.expressions.Length;
import org.apache.spark.sql.catalyst.expressions.LessThan;
import org.apache.spark.sql.catalyst.expressions.LessThanOrEqual;
import org.apache.spark.sql.catalyst.expressions.Literal;
import org.apache.spark.sql.catalyst.expressions.Log;
import org.apache.spark.sql.catalyst.expressions.Log10;
import org.apache.spark.sql.catalyst.expressions.Lower;
import org.apache.spark.sql.catalyst.expressions.Multiply;
import org.apache.spark.sql.catalyst.expressions.Not;
import org.apache.spark.sql.catalyst.expressions.Or;
import org.apache.spark.sql.catalyst.expressions.Pow;
import org.apache.spark.sql.catalyst.expressions.RLike;
import org.apache.spark.sql.catalyst.expressions.RegExpReplace;
import org.apache.spark.sql.catalyst.expressions.Rint;
import org.apache.spark.sql.catalyst.expressions.Sqrt;
import org.apache.spark.sql.catalyst.expressions.StringTrim;
import org.apache.spark.sql.catalyst.expressions.Substring;
import org.apache.spark.sql.catalyst.expressions.Subtract;
import org.apache.spark.sql.catalyst.expressions.UnaryExpression;
import org.apache.spark.sql.catalyst.expressions.UnaryMinus;
import org.apache.spark.sql.catalyst.expressions.UnaryPositive;
import org.apache.spark.sql.catalyst.expressions.Upper;
import org.apache.spark.sql.types.Decimal;
import org.dmg.pmml.Apply;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.HasDataType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.ValueUtil;
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

			String function;

			if(expression instanceof And || expression instanceof Or){

				switch(symbol){
					case "&&":
						function = PMMLFunctions.AND;
						break;
					case "||":
						function = PMMLFunctions.OR;
						break;
					default:
						throw new IllegalArgumentException(formatMessage(binaryOperator));
				}
			} else

			if(expression instanceof Add || expression instanceof Divide || expression instanceof Multiply || expression instanceof Subtract){
				BinaryArithmetic binaryArithmetic = (BinaryArithmetic)binaryOperator;

				switch(symbol){
					case "+":
						function = PMMLFunctions.ADD;
						break;
					case "/":
						function = PMMLFunctions.DIVIDE;
						break;
					case "*":
						function = PMMLFunctions.MULTIPLY;
						break;
					case "-":
						function = PMMLFunctions.SUBTRACT;
						break;
					default:
						throw new IllegalArgumentException(formatMessage(binaryArithmetic));
				}
			} else

			if(expression instanceof EqualTo || expression instanceof GreaterThan || expression instanceof GreaterThanOrEqual || expression instanceof LessThan || expression instanceof LessThanOrEqual){
				BinaryComparison binaryComparison = (BinaryComparison)binaryOperator;

				switch(symbol){
					case "=":
						function = PMMLFunctions.EQUAL;
						break;
					case ">":
						function = PMMLFunctions.GREATERTHAN;
						break;
					case ">=":
						function = PMMLFunctions.GREATEROREQUAL;
						break;
					case "<":
						function = PMMLFunctions.LESSTHAN;
						break;
					case "<=":
						function = PMMLFunctions.LESSOREQUAL;
						break;
					default:
						throw new IllegalArgumentException(formatMessage(binaryComparison));
				}
			} else

			{
				throw new IllegalArgumentException(formatMessage(binaryOperator));
			}

			return PMMLUtil.createApply(function, translateInternal(left), translateInternal(right));
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

				Apply branchApply = PMMLUtil.createApply(PMMLFunctions.IF)
					.addExpressions(translateInternal(predicate))
					.addExpressions(translateInternal(value));

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
				throw new IllegalArgumentException(formatMessage(cast));
			}
		} else

		if(expression instanceof Concat){
			Concat concat = (Concat)expression;

			List<Expression> children = JavaConversions.seqAsJavaList(concat.children());

			Apply apply = PMMLUtil.createApply(PMMLFunctions.CONCAT);

			for(Expression child : children){
				apply.addExpressions(translateInternal(child));
			}

			return apply;
		} else

		if(expression instanceof If){
			If _if = (If)expression;

			Expression predicate = _if.predicate();

			Expression trueValue = _if.trueValue();
			Expression falseValue = _if.falseValue();

			return PMMLUtil.createApply(PMMLFunctions.IF, translateInternal(predicate))
				.addExpressions(translateInternal(trueValue), translateInternal(falseValue));
		} else

		if(expression instanceof In){
			In in = (In)expression;

			Expression value = in.value();

			List<Expression> elements = JavaConversions.seqAsJavaList(in.list());

			Apply apply = PMMLUtil.createApply(PMMLFunctions.ISIN, translateInternal(value));

			for(Expression element : elements){
				apply.addExpressions(translateInternal(element));
			}

			return apply;
		} else

		if(expression instanceof Length){
			Length length = (Length)expression;

			Expression child = length.child();

			return PMMLUtil.createApply(PMMLFunctions.STRINGLENGTH, translateInternal(child));
		} else

		if(expression instanceof Literal){
			Literal literal = (Literal)expression;

			Object value = literal.value();

			DataType dataType;

			// XXX
			if(value instanceof Decimal){
				Decimal decimal = (Decimal)value;

				dataType = DataType.STRING;

				value = decimal.toString();
			} else

			{
				dataType = DatasetUtil.translateDataType(literal.dataType());

				value = toSimpleObject(value);
			}

			return PMMLUtil.createConstant(value, dataType);
		} else

		if(expression instanceof Pow){
			Pow pow = (Pow)expression;

			Expression left = pow.left();
			Expression right = pow.right();

			return PMMLUtil.createApply(PMMLFunctions.POW)
				.addExpressions(translateInternal(left), translateInternal(right));
		} else

		if(expression instanceof RegExpReplace){
			RegExpReplace regexpReplace = (RegExpReplace)expression;

			Expression subject = regexpReplace.subject();
			Expression regexp = regexpReplace.regexp();
			Expression rep = regexpReplace.rep();

			return PMMLUtil.createApply(PMMLFunctions.REPLACE)
				.addExpressions(translateInternal(subject))
				.addExpressions(translateInternal(regexp), translateInternal(rep));
		} else

		if(expression instanceof RLike){
			RLike rlike = (RLike)expression;

			Expression left = rlike.left();
			Expression right = rlike.right();

			return PMMLUtil.createApply(PMMLFunctions.MATCHES)
				.addExpressions(translateInternal(left))
				.addExpressions(translateInternal(right));
		} else

		if(expression instanceof StringTrim){
			StringTrim stringTrim = (StringTrim)expression;

			Expression srcStr = stringTrim.srcStr();
			Option<Expression> trimStr = stringTrim.trimStr();
			if(trimStr.isDefined()){
				throw new IllegalArgumentException();
			}

			return PMMLUtil.createApply(PMMLFunctions.TRIMBLANKS, translateInternal(srcStr));
		} else

		if(expression instanceof Substring){
			Substring substring = (Substring)expression;

			Expression str = substring.str();
			Literal pos = (Literal)substring.pos();
			Literal len = (Literal)substring.len();

			int posValue = ValueUtil.asInt((Number)pos.value());
			if(posValue <= 0){
				throw new IllegalArgumentException("Expected absolute start position, got relative start position " + (pos));
			}

			int lenValue = ValueUtil.asInt((Number)len.value());

			// XXX
			lenValue = Math.min(lenValue, MAX_STRING_LENGTH);

			return PMMLUtil.createApply(PMMLFunctions.SUBSTRING)
				.addExpressions(translateInternal(str))
				.addExpressions(PMMLUtil.createConstant(posValue), PMMLUtil.createConstant(lenValue));
		} else

		if(expression instanceof UnaryExpression){
			UnaryExpression unaryExpression = (UnaryExpression)expression;

			Expression child = unaryExpression.child();

			if(expression instanceof Abs){
				return PMMLUtil.createApply(PMMLFunctions.ABS, translateInternal(child));
			} else

			if(expression instanceof Ceil){
				return PMMLUtil.createApply(PMMLFunctions.CEIL, translateInternal(child));
			} else

			if(expression instanceof Exp){
				return PMMLUtil.createApply(PMMLFunctions.EXP, translateInternal(child));
			} else

			if(expression instanceof Floor){
				return PMMLUtil.createApply(PMMLFunctions.FLOOR, translateInternal(child));
			} else

			if(expression instanceof Log){
				return PMMLUtil.createApply(PMMLFunctions.LN, translateInternal(child));
			} else

			if(expression instanceof Log10){
				return PMMLUtil.createApply(PMMLFunctions.LOG10, translateInternal(child));
			} else

			if(expression instanceof Lower){
				return PMMLUtil.createApply(PMMLFunctions.LOWERCASE, translateInternal(child));
			} else

			if(expression instanceof IsNotNull){
				return PMMLUtil.createApply(PMMLFunctions.ISNOTMISSING, translateInternal(child));
			} else

			if(expression instanceof IsNull){
				return PMMLUtil.createApply(PMMLFunctions.ISMISSING, translateInternal(child));
			} else

			if(expression instanceof Not){
				 return PMMLUtil.createApply(PMMLFunctions.NOT, translateInternal(child));
			} else

			if(expression instanceof Rint){
				return PMMLUtil.createApply(PMMLFunctions.RINT, translateInternal(child));
			} else

			if(expression instanceof Sqrt){
				return PMMLUtil.createApply(PMMLFunctions.SQRT, translateInternal(child));
			} else

			if(expression instanceof UnaryMinus){
				return PMMLUtil.toNegative(translateInternal(child));
			} else

			if(expression instanceof UnaryPositive){
				return translateInternal(child);
			} else

			if(expression instanceof Upper){
				return PMMLUtil.createApply(PMMLFunctions.UPPERCASE, translateInternal(child));
			} else

			{
				throw new IllegalArgumentException(formatMessage(unaryExpression));
			}
		} else

		{
			throw new IllegalArgumentException(formatMessage(expression));
		}
	}

	static
	private Object toSimpleObject(Object value){
		Class<?> clazz = value.getClass();

		if(!(ExpressionTranslator.javaLangPackage).equals(clazz.getPackage())){
			return value.toString();
		}

		return value;
	}

	static
	private String formatMessage(Expression expression){

		if(expression == null){
			return null;
		}

		return "Spark SQL function \'" + expression + "\' (class " + (expression.getClass()).getName() + ") is not supported";
	}

	private static final Package javaLangPackage = Package.getPackage("java.lang");

	private static final int MAX_STRING_LENGTH = 65536;
}