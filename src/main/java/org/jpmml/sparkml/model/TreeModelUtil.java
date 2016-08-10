/*
 * Copyright (c) 2016 Villu Ruusmann
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
package org.jpmml.sparkml.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.tree.CategoricalSplit;
import org.apache.spark.ml.tree.ContinuousSplit;
import org.apache.spark.ml.tree.DecisionTreeModel;
import org.apache.spark.ml.tree.InternalNode;
import org.apache.spark.ml.tree.LeafNode;
import org.apache.spark.ml.tree.Split;
import org.apache.spark.ml.tree.TreeEnsembleModel;
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator;
import org.dmg.pmml.Array;
import org.dmg.pmml.MiningFunctionType;
import org.dmg.pmml.Node;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.SimpleSetPredicate;
import org.dmg.pmml.TreeModel;
import org.dmg.pmml.True;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.VisitorAction;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ListFeature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.model.visitors.AbstractVisitor;
import org.jpmml.sparkml.BooleanFeature;

public class TreeModelUtil {

	private TreeModelUtil(){
	}

	static
	public TreeModel encodeDecisionTree(DecisionTreeModel model, Schema schema){
		org.apache.spark.ml.tree.Node node = model.rootNode();

		if(model instanceof DecisionTreeRegressionModel){
			return encodeTreeModel(MiningFunctionType.REGRESSION, node, schema);
		} else

		if(model instanceof DecisionTreeClassificationModel){
			return encodeTreeModel(MiningFunctionType.CLASSIFICATION, node, schema);
		}

		throw new IllegalArgumentException();
	}

	static
	public List<TreeModel> encodeDecisionTreeEnsemble(TreeEnsembleModel model, final Schema schema){
		Function<DecisionTreeModel, TreeModel> function = new Function<DecisionTreeModel, TreeModel>(){

			private Schema segmentSchema = schema.toAnonymousSchema();


			@Override
			public TreeModel apply(DecisionTreeModel model){
				return encodeDecisionTree(model, this.segmentSchema);
			}
		};

		List<TreeModel> treeModels = new ArrayList<>(Lists.transform(Arrays.asList(model.trees()), function));

		return treeModels;
	}

	static
	public TreeModel encodeTreeModel(MiningFunctionType miningFunction, org.apache.spark.ml.tree.Node node, Schema schema){
		Node root = encodeNode(miningFunction, node, schema)
			.setPredicate(new True());

		TreeModel treeModel = new TreeModel(miningFunction, ModelUtil.createMiningSchema(schema), root)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.BINARY_SPLIT);

		return treeModel;
	}

	static
	public void scalePredictions(final TreeModel treeModel, final double weight){

		if(ValueUtil.isOne(weight)){
			return;
		}

		Visitor visitor = new AbstractVisitor(){

			@Override
			public VisitorAction visit(Node node){
				double score = Double.parseDouble(node.getScore());

				node.setScore(ValueUtil.formatValue(score * weight));

				return super.visit(node);
			}
		};
		visitor.applyTo(treeModel);
	}

	static
	public Node encodeNode(MiningFunctionType miningFunction, org.apache.spark.ml.tree.Node node, Schema schema){

		if(node instanceof InternalNode){
			return encodeInternalNode(miningFunction, (InternalNode)node, schema);
		} else

		if(node instanceof LeafNode){
			return encodeLeafNode(miningFunction, (LeafNode)node, schema);
		}

		throw new IllegalArgumentException();
	}

	static
	private Node encodeInternalNode(MiningFunctionType miningFunction, InternalNode internalNode, Schema schema){
		Node result = createNode(miningFunction, internalNode, schema);

		Predicate[] predicates = encodeSplit(internalNode.split(), schema);

		Node leftChild = encodeNode(miningFunction, internalNode.leftChild(), schema)
			.setPredicate(predicates[0]);

		Node rightChild = encodeNode(miningFunction, internalNode.rightChild(), schema)
			.setPredicate(predicates[1]);

		result.addNodes(leftChild, rightChild);

		return result;
	}

	static
	private Node encodeLeafNode(MiningFunctionType miningFunction, LeafNode leafNode, Schema schema){
		Node result = createNode(miningFunction, leafNode, schema);

		return result;
	}

	static
	private Node createNode(MiningFunctionType miningFunction, org.apache.spark.ml.tree.Node node, Schema schema){
		Node result = new Node();

		switch(miningFunction){
			case REGRESSION:
				{
					String score = ValueUtil.formatValue(node.prediction());

					result.setScore(score);
				}
				break;
			case CLASSIFICATION:
				{
					List<String> targetCategories = schema.getTargetCategories();
					if(targetCategories == null){
						throw new IllegalArgumentException();
					}

					int index = ValueUtil.asInt(node.prediction());

					result.setScore(targetCategories.get(index));

					ImpurityCalculator impurityCalculator = node.impurityStats();

					result.setRecordCount((double)impurityCalculator.count());

					double[] stats = impurityCalculator.stats();
					for(int i = 0; i < stats.length; i++){

						if(stats[i] == 0d){
							continue;
						}

						ScoreDistribution scoreDistribution = new ScoreDistribution(targetCategories.get(i), stats[i]);

						result.addScoreDistributions(scoreDistribution);
					}
				}
				break;
			default:
				throw new UnsupportedOperationException();
		}

		return result;
	}

	static
	private Predicate[] encodeSplit(Split split, Schema schema){

		if(split instanceof ContinuousSplit){
			return encodeContinuousSplit((ContinuousSplit)split, schema);
		} else

		if(split instanceof CategoricalSplit){
			return encodeCategoricalSplit((CategoricalSplit)split, schema);
		}

		throw new IllegalArgumentException();
	}

	static
	private Predicate[] encodeContinuousSplit(ContinuousSplit continuousSplit, Schema schema){
		ContinuousFeature feature = (ContinuousFeature)schema.getFeature(continuousSplit.featureIndex());

		double threshold = continuousSplit.threshold();

		if(feature instanceof BooleanFeature){
			BooleanFeature booleanFeature = (BooleanFeature)feature;

			if(threshold != 0d){
				throw new IllegalArgumentException();
			}

			SimplePredicate leftPredicate = new SimplePredicate(feature.getName(), SimplePredicate.Operator.EQUAL)
				.setValue(booleanFeature.getValue(0));

			SimplePredicate rightPredicate = new SimplePredicate(feature.getName(), SimplePredicate.Operator.EQUAL)
				.setValue(booleanFeature.getValue(1));

			return new Predicate[]{leftPredicate, rightPredicate};
		} else

		{
			String value = ValueUtil.formatValue(threshold);

			SimplePredicate leftPredicate = new SimplePredicate(feature.getName(), SimplePredicate.Operator.LESS_OR_EQUAL)
				.setValue(value);

			SimplePredicate rightPredicate = new SimplePredicate(feature.getName(), SimplePredicate.Operator.GREATER_THAN)
				.setValue(value);

			return new Predicate[]{leftPredicate, rightPredicate};
		}
	}

	static
	private Predicate[] encodeCategoricalSplit(CategoricalSplit categoricalSplit, Schema schema){
		Feature feature = schema.getFeature(categoricalSplit.featureIndex());

		double[] leftCategories = categoricalSplit.leftCategories();
		double[] rightCategories = categoricalSplit.rightCategories();

		if(feature instanceof ListFeature){
			ListFeature listFeature = (ListFeature)feature;

			List<String> values = listFeature.getValues();
			if(values.size() != (leftCategories.length + rightCategories.length)){
				throw new IllegalArgumentException();
			}

			Predicate leftPredicate = createCategoricalPredicate(listFeature, leftCategories);

			Predicate rightPredicate = createCategoricalPredicate(listFeature, rightCategories);

			return new Predicate[]{leftPredicate, rightPredicate};
		} else

		if(feature instanceof BinaryFeature){
			BinaryFeature binaryFeature = (BinaryFeature)feature;

			SimplePredicate.Operator leftOperator;
			SimplePredicate.Operator rightOperator;

			if(Arrays.equals(TRUE, leftCategories) && Arrays.equals(FALSE, rightCategories)){
				leftOperator = SimplePredicate.Operator.EQUAL;
				rightOperator = SimplePredicate.Operator.NOT_EQUAL;
			} else

			if(Arrays.equals(FALSE, leftCategories) && Arrays.equals(TRUE, rightCategories)){
				leftOperator = SimplePredicate.Operator.NOT_EQUAL;
				rightOperator = SimplePredicate.Operator.EQUAL;
			} else

			{
				throw new IllegalArgumentException();
			}

			String value = ValueUtil.formatValue(binaryFeature.getValue());

			SimplePredicate leftPredicate = new SimplePredicate(binaryFeature.getName(), leftOperator)
				.setValue(value);

			SimplePredicate rightPredicate = new SimplePredicate(binaryFeature.getName(), rightOperator)
				.setValue(value);

			return new Predicate[]{leftPredicate, rightPredicate};
		}

		throw new IllegalArgumentException();
	}

	static
	private Predicate createCategoricalPredicate(ListFeature listFeature, double[] categories){
		List<String> values = new ArrayList<>();

		for(int i = 0; i < categories.length; i++){
			int index = ValueUtil.asInt(categories[i]);

			String value = listFeature.getValue(index);

			values.add(value);
		}

		if(values.size() == 1){
			String value = values.get(0);

			SimplePredicate simplePredicate = new SimplePredicate()
				.setField(listFeature.getName())
				.setOperator(SimplePredicate.Operator.EQUAL)
				.setValue(value);

			return simplePredicate;
		} else

		{
			Array array = new Array(Array.Type.INT, ValueUtil.formatArrayValue(values));

			SimpleSetPredicate simpleSetPredicate = new SimpleSetPredicate()
				.setField(listFeature.getName())
				.setBooleanOperator(SimpleSetPredicate.BooleanOperator.IS_IN)
				.setArray(array);

			return simpleSetPredicate;
		}
	}

	private static final double[] TRUE = {1.0d};
	private static final double[] FALSE = {0.0d};
}