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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.spark.ml.Model;
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
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.ScoreDistribution;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.TreeModelOptions;
import org.jpmml.sparkml.visitors.TreeModelCompactor;

public class TreeModelUtil {

	private TreeModelUtil(){
	}

	static
	public <M extends Model<M> & TreeEnsembleModel<T>, T extends Model<T> & DecisionTreeModel> List<TreeModel> encodeDecisionTreeEnsemble(M model, Schema schema){
		PredicateManager predicateManager = new PredicateManager();

		return encodeDecisionTreeEnsemble(model, predicateManager, schema);
	}

	static
	public <M extends Model<M> & TreeEnsembleModel<T>, T extends Model<T> & DecisionTreeModel> List<TreeModel> encodeDecisionTreeEnsemble(M model, PredicateManager predicateManager, Schema schema){
		Schema segmentSchema = schema.toAnonymousSchema();

		List<TreeModel> treeModels = new ArrayList<>();

		T[] trees = model.trees();
		for(T tree : trees){
			TreeModel treeModel = encodeDecisionTree(tree, predicateManager, segmentSchema);

			treeModels.add(treeModel);
		}

		return treeModels;
	}

	static
	public <M extends Model<M> & DecisionTreeModel> TreeModel encodeDecisionTree(M model, Schema schema){
		PredicateManager predicateManager = new PredicateManager();

		return encodeDecisionTree(model, predicateManager, schema);
	}

	static
	public <M extends Model<M> & DecisionTreeModel> TreeModel encodeDecisionTree(M model, PredicateManager predicateManager, Schema schema){
		org.apache.spark.ml.tree.Node node = model.rootNode();

		if(model instanceof DecisionTreeRegressionModel){
			return encodeTreeModel(node, predicateManager, MiningFunction.REGRESSION, schema);
		} else

		if(model instanceof DecisionTreeClassificationModel){
			return encodeTreeModel(node, predicateManager, MiningFunction.CLASSIFICATION, schema);
		}

		throw new IllegalArgumentException();
	}

	static
	public TreeModel encodeTreeModel(org.apache.spark.ml.tree.Node node, PredicateManager predicateManager, MiningFunction miningFunction, Schema schema){
		Node root = encodeNode(node, predicateManager, Collections.<FieldName, Set<String>>emptyMap(), miningFunction, schema)
			.setPredicate(new True());

		TreeModel treeModel = new TreeModel(miningFunction, ModelUtil.createMiningSchema(schema.getLabel()), root)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.BINARY_SPLIT);

		String compact = TreeModelOptions.COMPACT;
		if(compact != null && Boolean.valueOf(compact)){
			Visitor visitor = new TreeModelCompactor();

			visitor.applyTo(treeModel);
		}

		return treeModel;
	}

	static
	public Node encodeNode(org.apache.spark.ml.tree.Node node, PredicateManager predicateManager, Map<FieldName, Set<String>> parentFieldValues, MiningFunction miningFunction, Schema schema){

		if(node instanceof InternalNode){
			InternalNode internalNode = (InternalNode)node;

			Map<FieldName, Set<String>> leftFieldValues = parentFieldValues;
			Map<FieldName, Set<String>> rightFieldValues = parentFieldValues;

			Predicate leftPredicate;
			Predicate rightPredicate;

			Split split = internalNode.split();

			Feature feature = schema.getFeature(split.featureIndex());

			if(split instanceof ContinuousSplit){
				ContinuousSplit continuousSplit = (ContinuousSplit)split;

				double threshold = continuousSplit.threshold();

				if(feature instanceof BooleanFeature){
					BooleanFeature booleanFeature = (BooleanFeature)feature;

					if(threshold != 0d){
						throw new IllegalArgumentException();
					}

					leftPredicate = predicateManager.createSimplePredicate(booleanFeature, SimplePredicate.Operator.EQUAL, booleanFeature.getValue(0));
					rightPredicate = predicateManager.createSimplePredicate(booleanFeature, SimplePredicate.Operator.EQUAL, booleanFeature.getValue(1));
				} else

				{
					ContinuousFeature continuousFeature = feature.toContinuousFeature();

					String value = ValueUtil.formatValue(threshold);

					leftPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.LESS_OR_EQUAL, value);
					rightPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.GREATER_THAN, value);
				}
			} else

			if(split instanceof CategoricalSplit){
				CategoricalSplit categoricalSplit = (CategoricalSplit)split;

				double[] leftCategories = categoricalSplit.leftCategories();
				double[] rightCategories = categoricalSplit.rightCategories();

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

					leftPredicate = predicateManager.createSimplePredicate(binaryFeature, leftOperator, value);
					rightPredicate = predicateManager.createSimplePredicate(binaryFeature, rightOperator, value);
				} else

				if(feature instanceof CategoricalFeature){
					CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

					FieldName name = categoricalFeature.getName();

					List<String> values = categoricalFeature.getValues();
					if(values.size() != (leftCategories.length + rightCategories.length)){
						throw new IllegalArgumentException();
					}

					final
					Set<String> parentValues = parentFieldValues.get(name);

					com.google.common.base.Predicate<String> valueFilter = new com.google.common.base.Predicate<String>(){

						@Override
						public boolean apply(String value){

							if(parentValues != null){
								return parentValues.contains(value);
							}

							return true;
						}
					};

					List<String> leftValues = selectValues(values, leftCategories, valueFilter);
					List<String> rightValues = selectValues(values, rightCategories, valueFilter);

					leftFieldValues = new HashMap<>(parentFieldValues);
					leftFieldValues.put(name, new HashSet<>(leftValues));

					rightFieldValues = new HashMap<>(parentFieldValues);
					rightFieldValues.put(name, new HashSet<>(rightValues));

					leftPredicate = predicateManager.createSimpleSetPredicate(categoricalFeature, leftValues);
					rightPredicate = predicateManager.createSimpleSetPredicate(categoricalFeature, rightValues);
				} else

				{
					throw new IllegalArgumentException();
				}
			} else

			{
				throw new IllegalArgumentException();
			}

			Node result = new Node();

			Node leftChild = encodeNode(internalNode.leftChild(), predicateManager, leftFieldValues, miningFunction, schema)
				.setPredicate(leftPredicate);

			Node rightChild = encodeNode(internalNode.rightChild(), predicateManager, rightFieldValues, miningFunction, schema)
				.setPredicate(rightPredicate);

			result.addNodes(leftChild, rightChild);

			return result;
		} else

		if(node instanceof LeafNode){
			LeafNode leafNode = (LeafNode)node;

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
						CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

						int index = ValueUtil.asInt(node.prediction());

						result.setScore(categoricalLabel.getValue(index));

						ImpurityCalculator impurityCalculator = node.impurityStats();

						result.setRecordCount((double)impurityCalculator.count());

						double[] stats = impurityCalculator.stats();
						for(int i = 0; i < stats.length; i++){
							ScoreDistribution scoreDistribution = new ScoreDistribution(categoricalLabel.getValue(i), stats[i]);

							result.addScoreDistributions(scoreDistribution);
						}
					}
					break;
				default:
					throw new UnsupportedOperationException();
			}

			return result;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	private List<String> selectValues(List<String> values, double[] categories, com.google.common.base.Predicate<String> valueFilter){

		if(categories.length == 1){
			int index = ValueUtil.asInt(categories[0]);

			String value = values.get(index);

			if(valueFilter.apply(value)){
				return Collections.singletonList(value);
			}

			return Collections.emptyList();
		} else

		{
			List<String> result = new ArrayList<>(categories.length);

			for(int i = 0; i < categories.length; i++){
				int index = ValueUtil.asInt(categories[i]);

				String value = values.get(index);

				if(valueFilter.apply(value)){
					result.add(value);
				}
			}

			return result;
		}
	}

	private static final double[] TRUE = {1.0d};
	private static final double[] FALSE = {0.0d};
}