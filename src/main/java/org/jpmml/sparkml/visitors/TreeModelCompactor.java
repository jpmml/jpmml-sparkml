/*
 * Copyright (c) 2017 Villu Ruusmann
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
package org.jpmml.sparkml.visitors;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.HasFieldReference;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.SimpleSetPredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.visitors.AbstractTreeModelTransformer;

public class TreeModelCompactor extends AbstractTreeModelTransformer {

	private MiningFunction miningFunction = null;

	private Map<Node, SimpleSetPredicate> replacedPredicates = new IdentityHashMap<>();


	@Override
	public void enterNode(Node node){
		Object id = node.getId();
		Object score = node.getScore();

		if(id != null){
			throw new IllegalArgumentException();
		} // End if

		if(node.hasNodes()){
			List<Node> children = node.getNodes();

			if(children.size() != 2 || score != null){
				throw new IllegalArgumentException();
			}

			Node firstChild = children.get(0);
			Node secondChild = children.get(1);

			Predicate firstPredicate = firstChild.requirePredicate();
			Predicate secondPredicate = secondChild.requirePredicate();

			checkFieldReference(firstPredicate, secondPredicate);

			boolean update = true;

			if(hasOperator(firstPredicate, SimplePredicate.Operator.EQUAL) && hasOperator(secondPredicate, SimplePredicate.Operator.EQUAL)){
				update = isCategoricalField((SimplePredicate)firstPredicate);
			} else

			if(hasOperator(firstPredicate, SimplePredicate.Operator.NOT_EQUAL) && hasOperator(secondPredicate, SimplePredicate.Operator.EQUAL)){
				children = swapChildren(node);

				firstChild = children.get(0);
				secondChild = children.get(1);
			} else

			if(hasOperator(firstPredicate, SimplePredicate.Operator.EQUAL) && hasOperator(secondPredicate, SimplePredicate.Operator.NOT_EQUAL)){
				// Ignored
			} else

			if(hasOperator(firstPredicate, SimplePredicate.Operator.LESS_OR_EQUAL) && hasOperator(secondPredicate, SimplePredicate.Operator.GREATER_THAN)){
				// Ignored
			} else

			if(hasOperator(firstPredicate, SimplePredicate.Operator.EQUAL) && hasBooleanOperator(secondPredicate, SimpleSetPredicate.BooleanOperator.IS_IN)){
				addCategoricalField(secondChild);
			} else

			if(hasBooleanOperator(firstPredicate, SimpleSetPredicate.BooleanOperator.IS_IN) && hasOperator(secondPredicate, SimplePredicate.Operator.EQUAL)){
				children = swapChildren(node);

				firstChild = children.get(0);
				secondChild = children.get(1);

				addCategoricalField(secondChild);
			} else

			if(hasBooleanOperator(firstPredicate, SimpleSetPredicate.BooleanOperator.IS_IN) && hasBooleanOperator(secondPredicate, SimpleSetPredicate.BooleanOperator.IS_IN)){
				addCategoricalField(secondChild);
			} else

			{
				throw new IllegalArgumentException();
			} // End if

			if(update){
				secondChild.setPredicate(True.INSTANCE);
			}
		} else

		{
			if(score == null){
				throw new IllegalArgumentException();
			}
		}
	}

	@Override
	public void exitNode(Node node){
		Predicate predicate = node.requirePredicate();

		if(predicate instanceof True){
			Node parentNode = getParentNode();

			if(parentNode == null){
				return;
			} // End if

			if(this.miningFunction == MiningFunction.REGRESSION){
				initScore(parentNode, node);
				replaceChildWithGrandchildren(parentNode, node);
			} else

			if(this.miningFunction == MiningFunction.CLASSIFICATION){

				// Replace intermediate nodes, but not terminal nodes
				if(node.hasNodes()){
					replaceChildWithGrandchildren(parentNode, node);
				}
			} else

			{
				throw new IllegalArgumentException();
			}
		}
	}

	@Override
	public void enterTreeModel(TreeModel treeModel){
		TreeModel.MissingValueStrategy missingValueStrategy = treeModel.getMissingValueStrategy();
		TreeModel.NoTrueChildStrategy noTrueChildStrategy = treeModel.getNoTrueChildStrategy();
		TreeModel.SplitCharacteristic splitCharacteristic = treeModel.getSplitCharacteristic();

		if((missingValueStrategy != TreeModel.MissingValueStrategy.NONE) || (noTrueChildStrategy != TreeModel.NoTrueChildStrategy.RETURN_NULL_PREDICTION) || (splitCharacteristic != TreeModel.SplitCharacteristic.BINARY_SPLIT)){
			throw new IllegalArgumentException();
		}

		this.miningFunction = treeModel.requireMiningFunction();

		this.replacedPredicates.clear();
	}

	@Override
	public void exitTreeModel(TreeModel treeModel){
		treeModel
			.setMissingValueStrategy(TreeModel.MissingValueStrategy.NULL_PREDICTION)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.MULTI_SPLIT);

		switch(this.miningFunction){
			case REGRESSION:
				treeModel.setNoTrueChildStrategy(TreeModel.NoTrueChildStrategy.RETURN_LAST_PREDICTION);
				break;
			case CLASSIFICATION:
				break;
			default:
				throw new IllegalArgumentException();
		}

		this.miningFunction = null;
	}

	private boolean isCategoricalField(HasFieldReference<?> hasFieldReference){
		String name = hasFieldReference.requireField();

		java.util.function.Predicate<Node> predicate = new java.util.function.Predicate<Node>(){

			@Override
			public boolean test(Node node){
				Predicate predicate = node.requirePredicate();

				if(predicate instanceof True){
					predicate = TreeModelCompactor.this.replacedPredicates.get(node);
				} // End if

				if(predicate instanceof SimpleSetPredicate){
					SimpleSetPredicate simpleSetPredicate = (SimpleSetPredicate)predicate;

					return hasFieldReference(simpleSetPredicate, name);
				}

				return false;
			}
		};

		Node ancestorNode = getAncestorNode(predicate);

		return (ancestorNode != null);
	}

	private void addCategoricalField(Node node){
		this.replacedPredicates.put(node, (SimpleSetPredicate)node.requirePredicate());
	}
}