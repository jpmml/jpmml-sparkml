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
import java.util.List;

import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.linalg.Vector;
import org.dmg.pmml.DataType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Entity;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.NormDiscrete;
import org.dmg.pmml.OpType;
import org.dmg.pmml.neural_network.Connection;
import org.dmg.pmml.neural_network.NeuralInput;
import org.dmg.pmml.neural_network.NeuralInputs;
import org.dmg.pmml.neural_network.NeuralLayer;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.dmg.pmml.neural_network.NeuralOutput;
import org.dmg.pmml.neural_network.NeuralOutputs;
import org.dmg.pmml.neural_network.Neuron;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.ClassificationModelConverter;

public class MultilayerPerceptronClassificationModelConverter extends ClassificationModelConverter<MultilayerPerceptronClassificationModel> {

	public MultilayerPerceptronClassificationModelConverter(MultilayerPerceptronClassificationModel model){
		super(model);
	}

	@Override
	public NeuralNetwork encodeModel(Schema schema){
		MultilayerPerceptronClassificationModel model = getTransformer();

		int[] layers = model.layers();
		Vector weights = model.weights();

		List<Feature> features = schema.getFeatures();
		if(features.size() != layers[0]){
			throw new IllegalArgumentException();
		}

		FieldName targetField = schema.getTargetField();

		List<String> targetCategories = schema.getTargetCategories();
		if(targetCategories.size() != layers[layers.length - 1]){
			throw new IllegalArgumentException();
		}

		NeuralInputs neuralInputs = new NeuralInputs();

		for(int column = 0; column < features.size(); column++){
			Feature feature = features.get(column);

			DerivedField derivedField = new DerivedField(OpType.CONTINUOUS, DataType.DOUBLE);

			if(feature instanceof ContinuousFeature){
				ContinuousFeature continuousFeature = (ContinuousFeature)feature;

				derivedField.setExpression(new FieldRef(continuousFeature.getName()));
			} else

			if(feature instanceof BinaryFeature){
				BinaryFeature binaryFeature = (BinaryFeature)feature;

				derivedField.setExpression(new NormDiscrete(binaryFeature.getName(), binaryFeature.getValue()));
			} else

			{
				throw new IllegalArgumentException();
			}

			NeuralInput neuralInput = new NeuralInput()
				.setId("0/" + String.valueOf(column + 1))
				.setDerivedField(derivedField);

			neuralInputs.addNeuralInputs(neuralInput);
		}

		List<? extends Entity> entities = neuralInputs.getNeuralInputs();

		List<NeuralLayer> neuralLayers = new ArrayList<>();

		int weightPos = 0;

		for(int i = 1; i < layers.length; i++){
			List<Neuron> neurons = new ArrayList<>();

			int rows = entities.size();
			int columns = layers[i];

			for(int column = 0; column < columns; column++){
				Neuron neuron = new Neuron()
					.setId(i + "/" + String.valueOf(column + 1));

				for(int row = 0; row < rows; row++){
					Entity entity = entities.get(row);

					Connection connection = new Connection()
						.setFrom(entity.getId())
						.setWeight(weights.apply(weightPos + (row * columns) + column));

					neuron.addConnections(connection);
				}

				neurons.add(neuron);
			}

			weightPos += (rows * columns);

			for(Neuron neuron : neurons){
				neuron.setBias(weights.apply(weightPos));

				weightPos++;
			}

			NeuralLayer neuralLayer = new NeuralLayer(neurons);

			if(i == (layers.length - 1)){
				neuralLayer
					.setActivationFunction(NeuralNetwork.ActivationFunction.IDENTITY)
					.setNormalizationMethod(NeuralNetwork.NormalizationMethod.SOFTMAX);
			}

			neuralLayers.add(neuralLayer);

			entities = neurons;
		}

		if(weightPos != weights.size()){
			throw new IllegalArgumentException();
		}

		NeuralOutputs neuralOutputs = new NeuralOutputs();

		for(int column = 0; column < targetCategories.size(); column++){
			String targetCategory = targetCategories.get(column);

			Entity entity = entities.get(column);

			DerivedField derivedField = new DerivedField(OpType.CATEGORICAL, DataType.STRING)
				.setExpression(new NormDiscrete(targetField, targetCategory));

			NeuralOutput neuralOutput = new NeuralOutput()
				.setOutputNeuron(entity.getId())
				.setDerivedField(derivedField);

			neuralOutputs.addNeuralOutputs(neuralOutput);
		}

		NeuralNetwork neuralNetwork = new NeuralNetwork(MiningFunction.CLASSIFICATION, NeuralNetwork.ActivationFunction.LOGISTIC, ModelUtil.createMiningSchema(schema), neuralInputs, neuralLayers)
			.setNeuralOutputs(neuralOutputs)
			.setOutput(ModelUtil.createProbabilityOutput(schema));

		return neuralNetwork;
	}
}