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
import java.util.List;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.tree.CategoricalSplit;
import org.apache.spark.ml.tree.ContinuousSplit;
import org.apache.spark.ml.tree.DecisionTreeModel;
import org.apache.spark.ml.tree.Split;
import org.apache.spark.ml.tree.TreeEnsembleModel;
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Predicate;
import org.dmg.pmml.SimplePredicate;
import org.dmg.pmml.True;
import org.dmg.pmml.Visitor;
import org.dmg.pmml.tree.BranchNode;
import org.dmg.pmml.tree.ClassifierNode;
import org.dmg.pmml.tree.LeafNode;
import org.dmg.pmml.tree.Node;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.BooleanFeature;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.CategoryManager;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PredicateManager;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ScoreDistributionManager;
import org.jpmml.converter.ValueUtil;
import org.jpmml.sparkml.ModelConverter;
import org.jpmml.sparkml.visitors.TreeModelCompactor;

public class TreeModelUtil {

	private TreeModelUtil(){
	}

	static
	public <C extends ModelConverter<? extends M> & HasTreeOptions, M extends Model<M> & DecisionTreeModel> TreeModel encodeDecisionTree(C converter, Schema schema){
		PredicateManager predicateManager = new PredicateManager();
		ScoreDistributionManager scoreDistributionManager = new ScoreDistributionManager();

		return encodeDecisionTree(converter, predicateManager, scoreDistributionManager, schema);
	}

	static
	public <C extends ModelConverter<? extends M> & HasTreeOptions, M extends Model<M> & DecisionTreeModel> TreeModel encodeDecisionTree(C converter, PredicateManager predicateManager, ScoreDistributionManager scoreDistributionManager, Schema schema){
		return encodeDecisionTree(converter, converter.getTransformer(), predicateManager, scoreDistributionManager, schema);
	}

	static
	public <C extends ModelConverter<? extends M> & HasTreeOptions, M extends Model<M> & TreeEnsembleModel<T>, T extends Model<T> & DecisionTreeModel> List<TreeModel> encodeDecisionTreeEnsemble(C converter, Schema schema){
		PredicateManager predicateManager = new PredicateManager();
		ScoreDistributionManager scoreDistributionManager = new ScoreDistributionManager();

		return encodeDecisionTreeEnsemble(converter, predicateManager, scoreDistributionManager, schema);
	}

	static
	public <C extends ModelConverter<? extends M> & HasTreeOptions, M extends Model<M> & TreeEnsembleModel<T>, T extends Model<T> & DecisionTreeModel> List<TreeModel> encodeDecisionTreeEnsemble(C converter, PredicateManager predicateManager, ScoreDistributionManager scoreDistributionManager, Schema schema){
		M model = converter.getTransformer();

		Schema segmentSchema = schema.toAnonymousSchema();

		List<TreeModel> treeModels = new ArrayList<>();

		T[] trees = model.trees();
		for(T tree : trees){
			TreeModel treeModel = encodeDecisionTree(converter, tree, predicateManager, scoreDistributionManager, segmentSchema);

			treeModels.add(treeModel);
		}

		return treeModels;
	}

	static
	private <M extends Model<M> & DecisionTreeModel> TreeModel encodeDecisionTree(ModelConverter<?> converter, M model, PredicateManager predicateManager, ScoreDistributionManager scoreDistributionManager, Schema schema){
		TreeModel treeModel;

		if(model instanceof DecisionTreeRegressionModel){
			ScoreEncoder scoreEncoder = new ScoreEncoder(){

				@Override
				public Node encode(Node node, org.apache.spark.ml.tree.LeafNode leafNode){
					node.setScore(leafNode.prediction());

					return node;
				}
			};

			treeModel = encodeTreeModel(MiningFunction.REGRESSION, scoreEncoder, model, predicateManager, schema);
		} else

		if(model instanceof DecisionTreeClassificationModel){
			ScoreEncoder scoreEncoder = new ScoreEncoder(){

				private CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();


				@Override
				public Node encode(Node node, org.apache.spark.ml.tree.LeafNode leafNode){
					node = new ClassifierNode(null, node.requirePredicate());

					int index = ValueUtil.asInt(leafNode.prediction());

					node.setScore(this.categoricalLabel.getValue(index));

					ImpurityCalculator impurityCalculator = leafNode.impurityStats();

					node.setRecordCount(ValueUtil.narrow(impurityCalculator.count()));

					scoreDistributionManager.addScoreDistributions(node, this.categoricalLabel.getValues(), impurityCalculator.stats());

					return node;
				}
			};

			treeModel = encodeTreeModel(MiningFunction.CLASSIFICATION, scoreEncoder, model, predicateManager, schema);
		} else

		{
			throw new IllegalArgumentException();
		}

		Boolean compact = (Boolean)converter.getOption(HasTreeOptions.OPTION_COMPACT, Boolean.TRUE);
		if(compact != null && compact){
			Visitor visitor = new TreeModelCompactor();

			visitor.applyTo(treeModel);
		}

		return treeModel;
	}

	static
	private <M extends Model<M> & DecisionTreeModel> TreeModel encodeTreeModel(MiningFunction miningFunction, ScoreEncoder scoreEncoder, M model, PredicateManager predicateManager, Schema schema){
		Node root = encodeNode(True.INSTANCE, scoreEncoder, model.rootNode(), predicateManager, new CategoryManager(), schema);

		TreeModel treeModel = new TreeModel(miningFunction, ModelUtil.createMiningSchema(schema.getLabel()), root)
			.setSplitCharacteristic(TreeModel.SplitCharacteristic.BINARY_SPLIT);

		return treeModel;
	}

	static
	private Node encodeNode(Predicate predicate, ScoreEncoder scoreEncoder, org.apache.spark.ml.tree.Node sparkNode, PredicateManager predicateManager, CategoryManager categoryManager, Schema schema){

		if(sparkNode instanceof org.apache.spark.ml.tree.LeafNode){
			org.apache.spark.ml.tree.LeafNode leafNode = (org.apache.spark.ml.tree.LeafNode)sparkNode;

			Node result = new LeafNode(null, predicate);

			return scoreEncoder.encode(result, leafNode);
		} else

		if(sparkNode instanceof org.apache.spark.ml.tree.InternalNode){
			org.apache.spark.ml.tree.InternalNode internalNode = (org.apache.spark.ml.tree.InternalNode)sparkNode;

			CategoryManager leftCategoryManager = categoryManager;
			CategoryManager rightCategoryManager = categoryManager;

			Predicate leftPredicate;
			Predicate rightPredicate;

			Split split = internalNode.split();

			Feature feature = schema.getFeature(split.featureIndex());

			if(split instanceof ContinuousSplit){
				ContinuousSplit continuousSplit = (ContinuousSplit)split;

				Double threshold = continuousSplit.threshold();

				if(feature instanceof BooleanFeature){
					BooleanFeature booleanFeature = (BooleanFeature)feature;

					if(threshold != 0.5d){
						throw new IllegalArgumentException("Invalid split threshold value " + threshold + " for a boolean feature");
					}

					leftPredicate = predicateManager.createSimplePredicate(booleanFeature, SimplePredicate.Operator.EQUAL, booleanFeature.getValue(0));
					rightPredicate = predicateManager.createSimplePredicate(booleanFeature, SimplePredicate.Operator.EQUAL, booleanFeature.getValue(1));
				} else

				{
					ContinuousFeature continuousFeature = feature.toContinuousFeature();

					DataType dataType = continuousFeature.getDataType();
					switch(dataType){
						case INTEGER:
							threshold = Math.floor(threshold);
							break;
						default:
							break;
					}

					leftPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.LESS_OR_EQUAL, threshold);
					rightPredicate = predicateManager.createSimplePredicate(continuousFeature, SimplePredicate.Operator.GREATER_THAN, threshold);
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

					Object value = binaryFeature.getValue();

					leftPredicate = predicateManager.createSimplePredicate(binaryFeature, leftOperator, value);
					rightPredicate = predicateManager.createSimplePredicate(binaryFeature, rightOperator, value);
				} else

				if(feature instanceof CategoricalFeature){
					CategoricalFeature categoricalFeature = (CategoricalFeature)feature;

					String name = categoricalFeature.getName();

					List<?> values = categoricalFeature.getValues();
					if(values.size() != (leftCategories.length + rightCategories.length)){
						throw new IllegalArgumentException();
					}

					java.util.function.Predicate<Object> valueFilter = categoryManager.getValueFilter(name);

					List<Object> leftValues = selectValues(values, leftCategories, valueFilter);
					List<Object> rightValues = selectValues(values, rightCategories, valueFilter);

					leftCategoryManager = categoryManager.fork(name, leftValues);
					rightCategoryManager = categoryManager.fork(name, rightValues);

					leftPredicate = predicateManager.createPredicate(categoricalFeature, leftValues);
					rightPredicate = predicateManager.createPredicate(categoricalFeature, rightValues);
				} else

				{
					throw new IllegalArgumentException();
				}
			} else

			{
				throw new IllegalArgumentException();
			}

			Node leftChild = encodeNode(leftPredicate, scoreEncoder, internalNode.leftChild(), predicateManager, leftCategoryManager, schema);
			Node rightChild = encodeNode(rightPredicate, scoreEncoder, internalNode.rightChild(), predicateManager, rightCategoryManager, schema);

			Node result = new BranchNode(null, predicate)
				.addNodes(leftChild, rightChild);

			return result;
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	private List<Object> selectValues(List<?> values, double[] categories, java.util.function.Predicate<Object> valueFilter){

		if(categories.length == 1){
			int index = ValueUtil.asInt(categories[0]);

			Object value = values.get(index);

			if(valueFilter.test(value)){
				return Collections.singletonList(value);
			}

			return Collections.emptyList();
		} else

		{
			List<Object> result = new ArrayList<>(categories.length);

			for(int i = 0; i < categories.length; i++){
				int index = ValueUtil.asInt(categories[i]);

				Object value = values.get(index);

				if(valueFilter.test(value)){
					result.add(value);
				}
			}

			return result;
		}
	}

	interface ScoreEncoder {

		Node encode(Node node, org.apache.spark.ml.tree.LeafNode leafNode);
	}

	private static final double[] TRUE = {1.0d};
	private static final double[] FALSE = {0.0d};
}