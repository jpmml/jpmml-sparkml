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

import java.util.Objects;

import org.dmg.pmml.Expression;
import org.dmg.pmml.HasExpression;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.model.visitors.ExpressionFilterer;

public class AliasExpression extends Expression implements HasExpression<AliasExpression> {

	private String name = null;

	private Expression expression = null;


	public AliasExpression(String name, Expression expression){
		setName(name);
		setExpression(expression);
	}

	public String getName(){
		return this.name;
	}

	public AliasExpression setName(String name){
		this.name = Objects.requireNonNull(name);

		return this;
	}

	@Override
	public Expression requireExpression(){

		if(this.expression == null){
			throw new IllegalStateException();
		}

		return this.expression;
	}

	@Override
	public Expression getExpression(){
		return this.expression;
	}

	@Override
	public AliasExpression setExpression(Expression expression){
		this.expression = Objects.requireNonNull(expression);

		return this;
	}

	@Override
	public VisitorAction accept(Visitor visitor){
		VisitorAction status = visitor.visit(this);

		if(status == VisitorAction.CONTINUE){
			visitor.pushParent(this);

			status = PMMLObject.traverse(visitor, getExpression());

			visitor.popParent();
		} // End if

		if(status == VisitorAction.TERMINATE){
			return VisitorAction.TERMINATE;
		}

		return VisitorAction.CONTINUE;
	}

	static
	public Expression unwrap(Expression expression){
		expression = unwrapInternal(expression);

		ExpressionFilterer filterer = new ExpressionFilterer(){

			@Override
			public Expression filter(Expression expression){
				return unwrapInternal(expression);
			}
		};
		filterer.applyTo(expression);

		return expression;
	}

	static
	private Expression unwrapInternal(Expression expression){

		while(expression instanceof AliasExpression){
			AliasExpression aliasExpression = (AliasExpression)expression;

			expression = aliasExpression.getExpression();
		}

		return expression;
	}
}