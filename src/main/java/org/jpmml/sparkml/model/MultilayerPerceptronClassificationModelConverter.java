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
import org.apache.spark.ml.param.shared.HasProbabilityCol;
import org.dmg.pmml.DataType;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.neural_network.NeuralEntity;
import org.dmg.pmml.neural_network.NeuralInputs;
import org.dmg.pmml.neural_network.NeuralLayer;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.dmg.pmml.neural_network.Neuron;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.neural_network.NeuralNetworkUtil;
import org.jpmml.sparkml.ClassificationModelConverter;
import org.jpmml.sparkml.SparkMLEncoder;

public class MultilayerPerceptronClassificationModelConverter extends ClassificationModelConverter<MultilayerPerceptronClassificationModel> {

	public MultilayerPerceptronClassificationModelConverter(MultilayerPerceptronClassificationModel model){
		super(model);
	}

	@Override
	public List<OutputField> registerOutputFields(Label label, SparkMLEncoder encoder){
		MultilayerPerceptronClassificationModel model = getTransformer();

		List<OutputField> result = super.registerOutputFields(label, encoder);

		if(!(model instanceof HasProbabilityCol)){
			CategoricalLabel categoricalLabel = (CategoricalLabel)label;

			result = new ArrayList<>(result);
			result.addAll(ModelUtil.createProbabilityFields(DataType.DOUBLE, categoricalLabel.getValues()));
		}

		return result;
	}

	@Override
	public NeuralNetwork encodeModel(Schema schema){
		MultilayerPerceptronClassificationModel model = getTransformer();

		int[] layers = model.layers();
		Vector weights = model.weights();

		CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();
		List<? extends Feature> features = schema.getFeatures();

		SchemaUtil.checkSize(layers[layers.length - 1], categoricalLabel);
		SchemaUtil.checkSize(layers[0], features);

		NeuralInputs neuralInputs = NeuralNetworkUtil.createNeuralInputs(features, DataType.DOUBLE);

		List<? extends NeuralEntity> entities = neuralInputs.getNeuralInputs();

		List<NeuralLayer> neuralLayers = new ArrayList<>();

		int weightPos = 0;

		for(int layer = 1; layer < layers.length; layer++){
			NeuralLayer neuralLayer = new NeuralLayer();

			int rows = entities.size();
			int columns = layers[layer];

			List<List<Double>> weightMatrix = new ArrayList<>();

			for(int column = 0; column < columns; column++){
				List<Double> weightVector = new ArrayList<>();

				for(int row = 0; row < rows; row++){
					weightVector.add(weights.apply(weightPos + (row * columns) + column));
				}

				weightMatrix.add(weightVector);
			}

			weightPos += (rows * columns);

			for(int column = 0; column < columns; column++){
				List<Double> weightVector = weightMatrix.get(column);
				Double bias = weights.apply(weightPos);

				Neuron neuron = NeuralNetworkUtil.createNeuron(entities, weightVector, bias)
					.setId(String.valueOf(layer) + "/" + String.valueOf(column + 1));

				neuralLayer.addNeurons(neuron);

				weightPos++;
			}

			if(layer == (layers.length - 1)){
				neuralLayer
					.setActivationFunction(NeuralNetwork.ActivationFunction.IDENTITY)
					.setNormalizationMethod(NeuralNetwork.NormalizationMethod.SOFTMAX);
			}

			neuralLayers.add(neuralLayer);

			entities = neuralLayer.getNeurons();
		}

		if(weightPos != weights.size()){
			throw new IllegalArgumentException();
		}

		NeuralNetwork neuralNetwork = new NeuralNetwork(MiningFunction.CLASSIFICATION, NeuralNetwork.ActivationFunction.LOGISTIC, ModelUtil.createMiningSchema(categoricalLabel), neuralInputs, neuralLayers)
			.setNeuralOutputs(NeuralNetworkUtil.createClassificationNeuralOutputs(entities, categoricalLabel));

		return neuralNetwork;
	}
}