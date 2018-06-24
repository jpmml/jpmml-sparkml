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

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.function.Function;

import javax.xml.bind.JAXBException;

import com.google.common.collect.Iterables;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMML;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.MetroJAXBUtil;

public class PMMLBuilder {

	private StructType schema = null;

	private PipelineModel pipelineModel = null;

	private Map<String, Map<String, Object>> options = new LinkedHashMap<>();


	public PMMLBuilder(StructType schema, PipelineModel pipelineModel){
		ConverterFactory.checkVersion();

		setSchema(schema);
		setPipelineModel(pipelineModel);
	}

	public PMML build(){
		StructType schema = getSchema();
		PipelineModel pipelineModel = getPipelineModel();
		Map<String, ? extends Map<String, ?>> options = getOptions();

		ConverterFactory converterFactory = new ConverterFactory(options);

		SparkMLEncoder encoder = new SparkMLEncoder(schema, converterFactory);

		Map<FieldName, DerivedField> derivedFields = encoder.getDerivedFields();

		List<org.dmg.pmml.Model> models = new ArrayList<>();

		// Transformations preceding the last model
		List<FieldName> preProcessorNames = Collections.emptyList();

		Iterable<Transformer> transformers = getTransformers(pipelineModel);
		for(Transformer transformer : transformers){
			TransformerConverter<?> converter = converterFactory.newConverter(transformer);

			if(converter instanceof FeatureConverter){
				FeatureConverter<?> featureConverter = (FeatureConverter<?>)converter;

				featureConverter.registerFeatures(encoder);
			} else

			if(converter instanceof ModelConverter){
				ModelConverter<?> modelConverter = (ModelConverter<?>)converter;

				org.dmg.pmml.Model model = modelConverter.registerModel(encoder);

				models.add(model);

				preProcessorNames = new ArrayList<>(derivedFields.keySet());
			} else

			{
				throw new IllegalArgumentException("Expected a " + FeatureConverter.class.getName() + " or " + ModelConverter.class.getName() + " instance, got " + converter);
			}
		}

		// Transformations following the last model
		List<FieldName> postProcessorNames = new ArrayList<>(derivedFields.keySet());
		postProcessorNames.removeAll(preProcessorNames);

		org.dmg.pmml.Model rootModel;

		if(models.size() == 1){
			rootModel = Iterables.getOnlyElement(models);
		} else

		if(models.size() > 1){
			List<MiningField> targetMiningFields = new ArrayList<>();

			for(org.dmg.pmml.Model model : models){
				MiningSchema miningSchema = model.getMiningSchema();

				List<MiningField> miningFields = miningSchema.getMiningFields();
				for(MiningField miningField : miningFields){
					MiningField.UsageType usageType = miningField.getUsageType();

					switch(usageType){
						case PREDICTED:
						case TARGET:
							targetMiningFields.add(miningField);
							break;
						default:
							break;
					}
				}
			}

			MiningSchema miningSchema = new MiningSchema(targetMiningFields);

			MiningModel miningModel = MiningModelUtil.createModelChain(models, new Schema(null, Collections.emptyList()))
				.setMiningSchema(miningSchema);

			rootModel = miningModel;
		} else

		{
			throw new IllegalArgumentException("Expected a pipeline with one or more models, got a pipeline with zero models");
		}

		for(FieldName postProcessorName : postProcessorNames){
			DerivedField derivedField = derivedFields.get(postProcessorName);

			encoder.removeDerivedField(postProcessorName);

			Output output = ModelUtil.ensureOutput(rootModel);

			OutputField outputField = new OutputField(derivedField.getName(), derivedField.getDataType())
				.setOpType(derivedField.getOpType())
				.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
				.setExpression(derivedField.getExpression());

			output.addOutputFields(outputField);
		}

		PMML pmml = encoder.encodePMML(rootModel);

		return pmml;
	}

	public byte[] buildByteArray(){
		return buildByteArray(1024 * 1024);
	}

	private byte[] buildByteArray(int size){
		PMML pmml = build();

		ByteArrayOutputStream os = new ByteArrayOutputStream(size);

		try {
			MetroJAXBUtil.marshalPMML(pmml, os);
		} catch(JAXBException je){
			throw new RuntimeException(je);
		}

		return os.toByteArray();
	}

	public File buildFile(File file) throws IOException {
		PMML pmml = build();

		OutputStream os = new FileOutputStream(file);

		try {
			MetroJAXBUtil.marshalPMML(pmml, os);
		} catch(JAXBException je){
			throw new RuntimeException(je);
		} finally {
			os.close();
		}

		return file;
	}

	public PMMLBuilder putOption(PipelineStage pipelineStage, String key, Object value){
		return putOption(pipelineStage.uid(), key, value);
	}

	public PMMLBuilder putOptions(PipelineStage pipelineStage, Map<String, ?> map){
		return putOptions(pipelineStage.uid(), map);
	}

	public PMMLBuilder putOption(String uid, String key, Object value){
		return putOptions(uid, Collections.singletonMap(key, value));
	}

	public PMMLBuilder putOptions(String uid, Map<String, ?> map){
		Map<String, Map<String, Object>> options = getOptions();

		Map<String, Object> pipelineStageOptions = options.get(uid);
		if(pipelineStageOptions == null){
			pipelineStageOptions = new LinkedHashMap<>();

			options.put(uid, pipelineStageOptions);
		}

		pipelineStageOptions.putAll(map);

		return this;
	}

	public PMMLBuilder removeOptions(PipelineStage pipelineStage){
		return removeOptions(pipelineStage.uid());
	}

	public PMMLBuilder removeOptions(String uid){
		Map<String, Map<String, Object>> options = getOptions();

		options.remove(uid);

		return this;
	}

	public StructType getSchema(){
		return this.schema;
	}

	public PMMLBuilder setSchema(StructType schema){

		if(schema == null){
			throw new IllegalArgumentException();
		}

		this.schema = schema;

		return this;
	}

	public PipelineModel getPipelineModel(){
		return this.pipelineModel;
	}

	public PMMLBuilder setPipelineModel(PipelineModel pipelineModel){

		if(pipelineModel == null){
			throw new IllegalArgumentException();
		}

		this.pipelineModel = pipelineModel;

		return this;
	}

	public Map<String, Map<String, Object>> getOptions(){
		return this.options;
	}

	static
	private Iterable<Transformer> getTransformers(PipelineModel pipelineModel){
		List<Transformer> result = new ArrayList<>();
		result.add(pipelineModel);

		Function<Transformer, List<Transformer>> function = new Function<Transformer, List<Transformer>>(){

			@Override
			public List<Transformer> apply(Transformer transformer){

				if(transformer instanceof PipelineModel){
					PipelineModel pipelineModel = (PipelineModel)transformer;

					return Arrays.asList(pipelineModel.stages());
				} else

				if(transformer instanceof CrossValidatorModel){
					CrossValidatorModel crossValidatorModel = (CrossValidatorModel)transformer;

					return Collections.singletonList(crossValidatorModel.bestModel());
				} else

				if(transformer instanceof TrainValidationSplitModel){
					TrainValidationSplitModel trainValidationSplitModel = (TrainValidationSplitModel)transformer;

					return Collections.singletonList(trainValidationSplitModel.bestModel());
				}

				return null;
			}
		};

		while(true){
			boolean modified = false;

			ListIterator<Transformer> transformerIt = result.listIterator();
			while(transformerIt.hasNext()){
				Transformer transformer = transformerIt.next();

				List<Transformer> childTransformers = function.apply(transformer);
				if(childTransformers != null){
					transformerIt.remove();

					for(Transformer childTransformer : childTransformers){
						transformerIt.add(childTransformer);
					}

					modified = true;
				}
			}

			if(!modified){
				break;
			}
		}

		return result;
	}
}