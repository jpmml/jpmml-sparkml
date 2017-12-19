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

import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.PMMLObject;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.SimpleSetPredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.VisitorAction;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.model.visitors.AbstractVisitor;

public class TreeModelCompactor extends AbstractVisitor {

	private MiningFunction miningFunction = null;

	private Map<True, FieldName> categoricalFields = new IdentityHashMap<>();


	@Override
	public void pushParent(PMMLObject object){
		super.pushParent(object);

		if(object instanceof Node){
			handleNodePush((Node)object);
		} else

		if(object instanceof TreeModel){
			handleTreeModelPush((TreeModel)object);
		}
	}

	@Override
	public PMMLObject popParent(){
		PMMLObject object = super.popParent();

		if(object instanceof Node){
			handleNodePop((Node)object);
		} else

		if(object instanceof TreeModel){
			handleTreeModelPop((TreeModel)object);
		}

		return object;
	}

	@Override
	public VisitorAction visit(TreeModel treeModel){
		TreeModel.MissingValueStrategy missingValueStrategy = treeModel.getMissingValueStrategy();
		TreeModel.NoTrueChildStrategy noTrueChildStrategy = treeModel.getNoTrueChildStrategy();
		TreeModel.SplitCharacteristic splitCharacteristic = treeModel.getSplitCharacteristic();

		if(!(TreeModel.MissingValueStrategy.NONE).equals(missingValueStrategy) || !(TreeModel.NoTrueChildStrategy.RETURN_NULL_PREDICTION).equals(noTrueChildStrategy) || !(TreeModel.SplitCharacteristic.BINARY_SPLIT).equals(splitCharacteristic)){
			throw new IllegalArgumentException();
		}

		treeModel
			.setMissingValueStrategy(TreeModel.MissingValueStrategy.NULL_PREDICTION)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.MULTI_SPLIT);

		MiningFunction miningFunction = treeModel.getMiningFunction();
		switch(miningFunction){
			case REGRESSION:
				treeModel.setNoTrueChildStrategy(TreeModel.NoTrueChildStrategy.RETURN_LAST_PREDICTION);
				break;
			case CLASSIFICATION:
				break;
			default:
				throw new IllegalArgumentException();
		}

		return super.visit(treeModel);
	}

	private void handleNodePush(Node node){
		String id = node.getId();
		String score = node.getScore();

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

			Predicate firstPredicate = firstChild.getPredicate();
			Predicate secondPredicate = secondChild.getPredicate();

			predicate:
			if(firstPredicate instanceof SimplePredicate && secondPredicate instanceof SimplePredicate){
				SimplePredicate firstSimplePredicate = (SimplePredicate)firstPredicate;
				SimplePredicate secondSimplePredicate = (SimplePredicate)secondPredicate;

				SimplePredicate.Operator firstOperator = firstSimplePredicate.getOperator();
				SimplePredicate.Operator secondOperator = secondSimplePredicate.getOperator();

				if(!(firstSimplePredicate.getField()).equals(secondSimplePredicate.getField())){
					throw new IllegalArgumentException();
				} // End if

				if((SimplePredicate.Operator.EQUAL).equals(firstOperator) && (SimplePredicate.Operator.EQUAL).equals(secondOperator)){

					if(!isCategoricalField(firstSimplePredicate.getField())){
						break predicate;
					}

					secondChild.setPredicate(new True());
				} else

				{
					if(!(firstSimplePredicate.getValue()).equals(secondSimplePredicate.getValue())){
						throw new IllegalArgumentException();
					} // End if

					if((SimplePredicate.Operator.NOT_EQUAL).equals(firstOperator) && (SimplePredicate.Operator.EQUAL).equals(secondOperator)){
						swapChildren(children);

						firstChild = children.get(0);
						secondChild = children.get(1);
					} else

					if((SimplePredicate.Operator.EQUAL).equals(firstOperator) && (SimplePredicate.Operator.NOT_EQUAL).equals(secondOperator)){
						// Ignored
					} else

					if((SimplePredicate.Operator.LESS_OR_EQUAL).equals(firstOperator) && (SimplePredicate.Operator.GREATER_THAN).equals(secondOperator)){
						// Ignored
					} else

					{
						throw new IllegalArgumentException();
					}

					secondChild.setPredicate(new True());
				}
			} else

			if(firstPredicate instanceof SimplePredicate && secondPredicate instanceof SimpleSetPredicate){
				SimplePredicate simplePredicate = (SimplePredicate)firstPredicate;
				SimpleSetPredicate simpleSetPredicate = (SimpleSetPredicate)secondPredicate;

				if(!(simplePredicate.getField()).equals(simpleSetPredicate.getField()) || !(SimplePredicate.Operator.EQUAL).equals(simplePredicate.getOperator()) || !(SimpleSetPredicate.BooleanOperator.IS_IN).equals(simpleSetPredicate.getBooleanOperator())){
					throw new IllegalArgumentException();
				}

				secondChild.setPredicate(addCategoricalField(simpleSetPredicate));
			} else

			if(firstPredicate instanceof SimpleSetPredicate && secondPredicate instanceof SimplePredicate){
				SimpleSetPredicate simpleSetPredicate = (SimpleSetPredicate)firstPredicate;
				SimplePredicate simplePredicate = (SimplePredicate)secondPredicate;

				if(!(simpleSetPredicate.getField()).equals(simplePredicate.getField()) || !(SimpleSetPredicate.BooleanOperator.IS_IN).equals(simpleSetPredicate.getBooleanOperator()) || !(SimplePredicate.Operator.EQUAL).equals(simplePredicate.getOperator())){
					throw new IllegalArgumentException();
				}

				swapChildren(children);

				firstChild = children.get(0);
				secondChild = children.get(1);

				secondChild.setPredicate(addCategoricalField(simpleSetPredicate));
			} else

			if(firstPredicate instanceof SimpleSetPredicate && secondPredicate instanceof SimpleSetPredicate){
				SimpleSetPredicate firstSimpleSetPredicate = (SimpleSetPredicate)firstPredicate;
				SimpleSetPredicate secondSimpleSetPredicate = (SimpleSetPredicate)secondPredicate;

				if(!(firstSimpleSetPredicate.getField()).equals(secondSimpleSetPredicate.getField()) || !(SimpleSetPredicate.BooleanOperator.IS_IN).equals(firstSimpleSetPredicate.getBooleanOperator()) || !(SimpleSetPredicate.BooleanOperator.IS_IN).equals(secondSimpleSetPredicate.getBooleanOperator())){
					throw new IllegalArgumentException();
				}

				secondChild.setPredicate(addCategoricalField(secondSimpleSetPredicate));
			} else

			{
				throw new IllegalArgumentException();
			}
		} else

		{
			if(score == null){
				throw new IllegalArgumentException();
			}
		}
	}

	private void handleNodePop(Node node){
		String score = node.getScore();
		Predicate predicate = node.getPredicate();

		if(predicate instanceof True){
			Node parentNode = getParentNode();

			if(parentNode == null){
				return;
			}

			String parentScore = parentNode.getScore();
			if(parentScore != null){
				throw new IllegalArgumentException();
			} // End if

			if((MiningFunction.REGRESSION).equals(this.miningFunction)){
				parentNode.setScore(score);

				List<Node> parentChildren = parentNode.getNodes();

				boolean success = parentChildren.remove(node);
				if(!success){
					throw new IllegalArgumentException();
				} // End if

				if(node.hasNodes()){
					List<Node> children = node.getNodes();

					parentChildren.addAll(children);
				}
			} else

			if((MiningFunction.CLASSIFICATION).equals(this.miningFunction)){

				if(node.hasNodes()){
					List<Node> parentChildren = parentNode.getNodes();

					boolean success = parentChildren.remove(node);
					if(!success){
						throw new IllegalArgumentException();
					}

					List<Node> children = node.getNodes();

					parentChildren.addAll(children);
				}
			} else

			{
				throw new IllegalArgumentException();
			}
		}
	}

	private void handleTreeModelPush(TreeModel treeModel){
		this.miningFunction = treeModel.getMiningFunction();

		this.categoricalFields.clear();
	}

	private void handleTreeModelPop(TreeModel treeModel){
		this.miningFunction = null;
	}

	private boolean isCategoricalField(FieldName name){
		Deque<PMMLObject> parents = getParents();

		for(PMMLObject parent : parents){

			if(parent instanceof Node){
				Node node = (Node)parent;

				Predicate predicate = node.getPredicate();

				if(predicate instanceof SimpleSetPredicate){
					SimpleSetPredicate simpleSetPredicate = (SimpleSetPredicate)predicate;

					FieldName categoricalField = simpleSetPredicate.getField();

					if((name).equals(categoricalField)){
						return true;
					}
				} else

				if(predicate instanceof True){
					True truePredicate = (True)predicate;

					FieldName categoricalField = this.categoricalFields.get(truePredicate);

					if((name).equals(categoricalField)){
						return true;
					}
				}
			} else

			{
				return false;
			}
		}

		return false;
	}

	private True addCategoricalField(SimpleSetPredicate simpleSetPredicate){
		True truePredicate = new True();

		this.categoricalFields.put(truePredicate, simpleSetPredicate.getField());

		return truePredicate;
	}

	private Node getParentNode(){
		Deque<PMMLObject> parents = getParents();

		PMMLObject parent = parents.peekFirst();
		if(parent instanceof Node){
			return (Node)parent;
		}

		return null;
	}

	static
	private void swapChildren(List<Node> children){
		Node firstChild = children.remove(0);

		children.add(1, firstChild);
	}
}