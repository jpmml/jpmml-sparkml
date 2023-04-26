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
import java.util.Objects;
import java.util.function.Function;

import org.apache.spark.sql.catalyst.expressions.Abs;
import org.apache.spark.sql.catalyst.expressions.Acos;
import org.apache.spark.sql.catalyst.expressions.Add;
import org.apache.spark.sql.catalyst.expressions.Alias;
import org.apache.spark.sql.catalyst.expressions.And;
import org.apache.spark.sql.catalyst.expressions.Asin;
import org.apache.spark.sql.catalyst.expressions.Atan;
import org.apache.spark.sql.catalyst.expressions.AttributeReference;
import org.apache.spark.sql.catalyst.expressions.BinaryArithmetic;
import org.apache.spark.sql.catalyst.expressions.BinaryComparison;
import org.apache.spark.sql.catalyst.expressions.BinaryMathExpression;
import org.apache.spark.sql.catalyst.expressions.BinaryOperator;
import org.apache.spark.sql.catalyst.expressions.CaseWhen;
import org.apache.spark.sql.catalyst.expressions.Cast;
import org.apache.spark.sql.catalyst.expressions.Ceil;
import org.apache.spark.sql.catalyst.expressions.Concat;
import org.apache.spark.sql.catalyst.expressions.Cos;
import org.apache.spark.sql.catalyst.expressions.Cosh;
import org.apache.spark.sql.catalyst.expressions.Divide;
import org.apache.spark.sql.catalyst.expressions.EqualTo;
import org.apache.spark.sql.catalyst.expressions.Exp;
import org.apache.spark.sql.catalyst.expressions.Expm1;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.expressions.Floor;
import org.apache.spark.sql.catalyst.expressions.GreaterThan;
import org.apache.spark.sql.catalyst.expressions.GreaterThanOrEqual;
import org.apache.spark.sql.catalyst.expressions.Greatest;
import org.apache.spark.sql.catalyst.expressions.Hypot;
import org.apache.spark.sql.catalyst.expressions.If;
import org.apache.spark.sql.catalyst.expressions.In;
import org.apache.spark.sql.catalyst.expressions.IsNaN;
import org.apache.spark.sql.catalyst.expressions.IsNotNull;
import org.apache.spark.sql.catalyst.expressions.IsNull;
import org.apache.spark.sql.catalyst.expressions.Least;
import org.apache.spark.sql.catalyst.expressions.Length;
import org.apache.spark.sql.catalyst.expressions.LessThan;
import org.apache.spark.sql.catalyst.expressions.LessThanOrEqual;
import org.apache.spark.sql.catalyst.expressions.Literal;
import org.apache.spark.sql.catalyst.expressions.Log;
import org.apache.spark.sql.catalyst.expressions.Log10;
import org.apache.spark.sql.catalyst.expressions.Log1p;
import org.apache.spark.sql.catalyst.expressions.Lower;
import org.apache.spark.sql.catalyst.expressions.Multiply;
import org.apache.spark.sql.catalyst.expressions.Not;
import org.apache.spark.sql.catalyst.expressions.Or;
import org.apache.spark.sql.catalyst.expressions.Pow;
import org.apache.spark.sql.catalyst.expressions.RLike;
import org.apache.spark.sql.catalyst.expressions.RegExpReplace;
import org.apache.spark.sql.catalyst.expressions.Rint;
import org.apache.spark.sql.catalyst.expressions.Sin;
import org.apache.spark.sql.catalyst.expressions.Sinh;
import org.apache.spark.sql.catalyst.expressions.Sqrt;
import org.apache.spark.sql.catalyst.expressions.StringReplace;
import org.apache.spark.sql.catalyst.expressions.StringTrim;
import org.apache.spark.sql.catalyst.expressions.Substring;
import org.apache.spark.sql.catalyst.expressions.Subtract;
import org.apache.spark.sql.catalyst.expressions.Tan;
import org.apache.spark.sql.catalyst.expressions.Tanh;
import org.apache.spark.sql.catalyst.expressions.UnaryExpression;
import org.apache.spark.sql.catalyst.expressions.UnaryMinus;
import org.apache.spark.sql.catalyst.expressions.UnaryPositive;
import org.apache.spark.sql.catalyst.expressions.Upper;
import org.apache.spark.sql.types.Decimal;
import org.dmg.pmml.Apply;
import org.dmg.pmml.Constant;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.HasDataType;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMMLFunctions;
import org.jpmml.converter.FieldNameUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.TypeUtil;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.visitors.ExpressionCompactor;
import scala.Option;
import scala.Tuple2;
import scala.collection.JavaConversions;

public class ExpressionTranslator {

	private SparkMLEncoder encoder = null;


	private ExpressionTranslator(SparkMLEncoder encoder){
		setEncoder(encoder);
	}

	public SparkMLEncoder getEncoder(){
		return this.encoder;
	}

	private void setEncoder(SparkMLEncoder encoder){
		this.encoder = Objects.requireNonNull(encoder);
	}

	static
	public org.dmg.pmml.Expression translate(SparkMLEncoder encoder, Expression expression){
		return translate(encoder, expression, true);
	}

	static
	public org.dmg.pmml.Expression translate(SparkMLEncoder encoder, Expression expression, boolean compact){
		ExpressionTranslator expressionTranslator = new ExpressionTranslator(encoder);

		org.dmg.pmml.Expression pmmlExpression = expressionTranslator.translateInternal(expression);

		if(compact){
			ExpressionCompactor expressionCompactor = new ExpressionCompactor();

			expressionCompactor.applyTo(pmmlExpression);
		}

		return pmmlExpression;
	}

	private org.dmg.pmml.Expression translateInternal(Expression expression){
		SparkMLEncoder encoder = getEncoder();

		if(expression instanceof Alias){
			Alias alias = (Alias)expression;

			String name = alias.name();
			Expression child = alias.child();

			org.dmg.pmml.Expression pmmlExpression = translateInternal(child);

			return new AliasExpression(name, pmmlExpression);
		} // End if

		if(expression instanceof AttributeReference){
			AttributeReference attributeReference = (AttributeReference)expression;

			String name = attributeReference.name();

			return new FieldRef(name);
		} else

		if(expression instanceof BinaryMathExpression){
			BinaryMathExpression binaryMathExpression = (BinaryMathExpression)expression;

			Expression left = binaryMathExpression.left();
			Expression right = binaryMathExpression.right();

			String function;

			if(binaryMathExpression instanceof Hypot){
				function = PMMLFunctions.HYPOT;
			} else

			if(binaryMathExpression instanceof Pow){
				function = PMMLFunctions.POW;
			} else

			{
				throw new IllegalArgumentException(formatMessage(binaryMathExpression));
			}

			return PMMLUtil.createApply(function, translateInternal(left), translateInternal(right));
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

				Apply branchApply = PMMLUtil.createApply(PMMLFunctions.IF,
					translateInternal(predicate),
					translateInternal(value)
				);

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

			org.dmg.pmml.Expression pmmlExpression = translateInternal(child);

			DataType dataType = DatasetUtil.translateDataType(cast.dataType());

			if(pmmlExpression instanceof HasDataType){
				HasDataType<?> hasDataType = (HasDataType<?>)pmmlExpression;

				hasDataType.setDataType(dataType);

				return pmmlExpression;
			} else

			{
				String name;

				if(pmmlExpression instanceof AliasExpression){
					AliasExpression aliasExpression = (AliasExpression)pmmlExpression;

					name = aliasExpression.getName();
				} else

				{
					name = FieldNameUtil.create(dataType, ExpressionUtil.format(child));
				}

				OpType opType = TypeUtil.getOpType(dataType);

				pmmlExpression = AliasExpression.unwrap(pmmlExpression);

				DerivedField derivedField = encoder.createDerivedField(name, opType, dataType, pmmlExpression);

				return new FieldRef(derivedField);
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

		if(expression instanceof Greatest){
			Greatest greatest = (Greatest)expression;

			List<Expression> children = JavaConversions.seqAsJavaList(greatest.children());

			Apply apply = PMMLUtil.createApply(PMMLFunctions.MAX);

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

			return PMMLUtil.createApply(PMMLFunctions.IF,
				translateInternal(predicate),
				translateInternal(trueValue),
				translateInternal(falseValue)
			);
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

		if(expression instanceof Least){
			Least least = (Least)expression;

			List<Expression> children = JavaConversions.seqAsJavaList(least.children());

			Apply apply = PMMLUtil.createApply(PMMLFunctions.MIN);

			for(Expression child : children){
				apply.addExpressions(translateInternal(child));
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
			if(value == null){
				return PMMLUtil.createMissingConstant();
			}

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

		if(expression instanceof RegExpReplace){
			RegExpReplace regexpReplace = (RegExpReplace)expression;

			Expression subject = regexpReplace.subject();
			Expression regexp = regexpReplace.regexp();
			Expression rep = regexpReplace.rep();

			return PMMLUtil.createApply(PMMLFunctions.REPLACE, translateInternal(subject), translateInternal(regexp), translateInternal(rep));
		} else

		if(expression instanceof RLike){
			RLike rlike = (RLike)expression;

			Expression left = rlike.left();
			Expression right = rlike.right();

			return PMMLUtil.createApply(PMMLFunctions.MATCHES, translateInternal(left), translateInternal(right));
		} else

		if(expression instanceof StringReplace){
			StringReplace stringReplace = (StringReplace)expression;

			Expression srcExpr = stringReplace.srcExpr();
			Expression searchExpr = stringReplace.searchExpr();
			Expression replaceExpr = stringReplace.replaceExpr();

			return PMMLUtil.createApply(PMMLFunctions.REPLACE, translateInternal(srcExpr), transformString(translateInternal(searchExpr), ExpressionTranslator::escapeSearchString), transformString(translateInternal(replaceExpr), ExpressionTranslator::escapeReplacementString));
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

			return PMMLUtil.createApply(PMMLFunctions.SUBSTRING, translateInternal(str), PMMLUtil.createConstant(posValue), PMMLUtil.createConstant(lenValue));
		} else

		if(expression instanceof UnaryExpression){
			UnaryExpression unaryExpression = (UnaryExpression)expression;

			Expression child = unaryExpression.child();

			if(expression instanceof Abs){
				return PMMLUtil.createApply(PMMLFunctions.ABS, translateInternal(child));
			} else

			if(expression instanceof Acos){
				return PMMLUtil.createApply(PMMLFunctions.ACOS, translateInternal(child));
			} else

			if(expression instanceof Asin){
				return PMMLUtil.createApply(PMMLFunctions.ASIN, translateInternal(child));
			} else

			if(expression instanceof Atan){
				return PMMLUtil.createApply(PMMLFunctions.ATAN, translateInternal(child));
			} else

			if(expression instanceof Ceil){
				return PMMLUtil.createApply(PMMLFunctions.CEIL, translateInternal(child));
			} else

			if(expression instanceof Cos){
				return PMMLUtil.createApply(PMMLFunctions.COS, translateInternal(child));
			} else

			if(expression instanceof Cosh){
				return PMMLUtil.createApply(PMMLFunctions.COSH, translateInternal(child));
			} else

			if(expression instanceof Exp){
				return PMMLUtil.createApply(PMMLFunctions.EXP, translateInternal(child));
			} else

			if(expression instanceof Expm1){
				return PMMLUtil.createApply(PMMLFunctions.EXPM1, translateInternal(child));
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

			if(expression instanceof Log1p){
				return PMMLUtil.createApply(PMMLFunctions.LN1P, translateInternal(child));
			} else

			if(expression instanceof Lower){
				return PMMLUtil.createApply(PMMLFunctions.LOWERCASE, translateInternal(child));
			} else

			if(expression instanceof IsNaN){
				// XXX
				return PMMLUtil.createApply(PMMLFunctions.ISNOTVALID, translateInternal(child));
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

			if(expression instanceof Sin){
				return PMMLUtil.createApply(PMMLFunctions.SIN, translateInternal(child));
			} else

			if(expression instanceof Sinh){
				return PMMLUtil.createApply(PMMLFunctions.SINH, translateInternal(child));
			} else

			if(expression instanceof Sqrt){
				return PMMLUtil.createApply(PMMLFunctions.SQRT, translateInternal(child));
			} else

			if(expression instanceof Tan){
				return PMMLUtil.createApply(PMMLFunctions.TAN, translateInternal(child));
			} else

			if(expression instanceof Tanh){
				return PMMLUtil.createApply(PMMLFunctions.TANH, translateInternal(child));
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
	private String escapeSearchString(String string){
		return escape(string, "<([{\\^-=$!|]})?*+.>");
	}

	static
	private String escapeReplacementString(String string){
		return escape(string, "\\$");
	}

	static
	private String escape(String string, String specialCharacters){
		StringBuilder sb = new StringBuilder();

		for(int i = 0; i < string.length(); i++){
			char c = string.charAt(i);

			if(specialCharacters.indexOf(c) > -1){
				sb.append('\\');
			}

			sb.append(c);
		}

		return sb.toString();
	}

	static
	private Constant transformString(org.dmg.pmml.Expression pmmlExpression, Function<String, String> function){
		Constant constant = (Constant)pmmlExpression;

		if(constant.getDataType() != DataType.STRING){
			throw new IllegalArgumentException();
		}

		constant.setValue(function.apply((String)constant.getValue()));

		return constant;
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

		return "Spark SQL function \'" + ExpressionUtil.format(expression) + "\' (class " + (expression.getClass()).getName() + ") is not supported";
	}

	private static final Package javaLangPackage = Package.getPackage("java.lang");

	private static final int MAX_STRING_LENGTH = 65536;
}