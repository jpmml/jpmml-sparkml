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
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.google.common.collect.Iterables;
import jakarta.xml.bind.JAXBException;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.apache.spark.ml.param.shared.HasProbabilityCol;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.MiningSchema;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.PMML;
import org.dmg.pmml.ResultFeature;
import org.dmg.pmml.VerificationField;
import org.dmg.pmml.mining.Segmentation;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.SchemaUtil;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.metro.MetroJAXBUtil;
import org.jpmml.sparkml.model.HasFeatureImportances;
import org.jpmml.sparkml.model.HasTreeOptions;

public class PMMLBuilder {

	private StructType schema = null;

	private PipelineModel pipelineModel = null;

	private Map<RegexKey, Map<String, Object>> options = new LinkedHashMap<>();

	private Verification verification = null;


	public PMMLBuilder(StructType schema, PipelineModel pipelineModel){
		setSchema(schema);
		setPipelineModel(pipelineModel);
	}

	public PMMLBuilder(StructType schema, PipelineStage pipelineStage){
		throw new IllegalArgumentException("Expected a fitted pipeline model (class " + PipelineModel.class.getName() + "), got a pipeline stage (" + (pipelineStage != null ? ("class " + (pipelineStage.getClass()).getName()) : null) + ")");
	}

	public PMML build(){
		StructType schema = getSchema();
		PipelineModel pipelineModel = getPipelineModel();
		Map<RegexKey, ? extends Map<String, ?>> options = getOptions();
		Verification verification = getVerification();

		ConverterFactory converterFactory = new ConverterFactory(options);

		SparkMLEncoder encoder = new SparkMLEncoder(schema, converterFactory);

		Map<String, DerivedField> derivedFields = encoder.getDerivedFields();

		List<org.dmg.pmml.Model> models = new ArrayList<>();

		List<String> predictionColumns = new ArrayList<>();
		List<String> probabilityColumns = new ArrayList<>();

		// Transformations preceding the last model
		List<String> preProcessorNames = Collections.emptyList();

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

				featureImportances:
				if(modelConverter instanceof HasFeatureImportances){
					HasFeatureImportances hasFeatureImportances = (HasFeatureImportances)modelConverter;

					Boolean estimateFeatureImportances = (Boolean)modelConverter.getOption(HasTreeOptions.OPTION_ESTIMATE_FEATURE_IMPORTANCES, Boolean.FALSE);
					if(!estimateFeatureImportances){
						break featureImportances;
					}

					List<Double> featureImportances = VectorUtil.toList(hasFeatureImportances.getFeatureImportances());

					List<Feature> features = modelConverter.getFeatures(encoder);

					SchemaUtil.checkSize(featureImportances.size(), features);

					for(int i = 0; i < featureImportances.size(); i++){
						Double featureImportance = featureImportances.get(i);
						Feature feature = features.get(i);

						encoder.addFeatureImportance(model, feature, featureImportance);
					}
				} // End if

				hasPredictionCol:
				if(transformer instanceof HasPredictionCol){
					HasPredictionCol hasPredictionCol = (HasPredictionCol)transformer;

					// XXX
					if((transformer instanceof GeneralizedLinearRegressionModel) && (model.requireMiningFunction() == MiningFunction.CLASSIFICATION)){
						break hasPredictionCol;
					}

					predictionColumns.add(hasPredictionCol.getPredictionCol());
				} // End if

				if(transformer instanceof HasProbabilityCol){
					HasProbabilityCol hasProbabilityCol = (HasProbabilityCol)transformer;

					probabilityColumns.add(hasProbabilityCol.getProbabilityCol());
				}

				preProcessorNames = new ArrayList<>(derivedFields.keySet());
			} else

			{
				throw new IllegalArgumentException("Expected a subclass of " + FeatureConverter.class.getName() + " or " + ModelConverter.class.getName() + ", got " + (converter != null ? ("class " + (converter.getClass()).getName()) : null));
			}
		}

		// Transformations following the last model
		List<String> postProcessorNames = new ArrayList<>(derivedFields.keySet());
		postProcessorNames.removeAll(preProcessorNames);

		org.dmg.pmml.Model model;

		if(models.size() == 0){
			model = null;
		} else

		if(models.size() == 1){
			model = Iterables.getOnlyElement(models);
		} else

		{
			model = MiningModelUtil.createModelChain(models, Segmentation.MissingPredictionTreatment.CONTINUE);
		} // End if

		if((model != null) && !postProcessorNames.isEmpty()){
			org.dmg.pmml.Model finalModel = MiningModelUtil.getFinalModel(model);

			Output output = ModelUtil.ensureOutput(finalModel);

			for(String postProcessorName : postProcessorNames){
				DerivedField derivedField = derivedFields.get(postProcessorName);

				encoder.removeDerivedField(postProcessorName);

				OutputField outputField = new OutputField(derivedField.requireName(), derivedField.requireOpType(), derivedField.requireDataType())
					.setResultFeature(ResultFeature.TRANSFORMED_VALUE)
					.setExpression(derivedField.requireExpression());

				output.addOutputFields(outputField);
			}
		}

		PMML pmml = encoder.encodePMML(model);

		if((model != null) && (!predictionColumns.isEmpty() || !probabilityColumns.isEmpty()) && (verification != null)){
			Dataset<Row> dataset = verification.getDataset();
			Dataset<Row> transformedDataset = verification.getTransformedDataset();
			Double precision = verification.getPrecision();
			Double zeroThreshold = verification.getZeroThreshold();

			List<String> inputColumns = new ArrayList<>();

			MiningSchema miningSchema = model.requireMiningSchema();

			List<MiningField> miningFields = miningSchema.getMiningFields();
			for(MiningField miningField : miningFields){
				MiningField.UsageType usageType = miningField.getUsageType();

				switch(usageType){
					case ACTIVE:
						String name = miningField.getName();

						inputColumns.add(name);
						break;
					default:
						break;
				}
			}

			Map<VerificationField, List<?>> data = new LinkedHashMap<>();

			for(String inputColumn : inputColumns){
				VerificationField verificationField = ModelUtil.createVerificationField(inputColumn);

				data.put(verificationField, getColumn(dataset, inputColumn));
			}

			for(String predictionColumn : predictionColumns){
				Feature feature = encoder.getOnlyFeature(predictionColumn);

				VerificationField verificationField = ModelUtil.createVerificationField(feature.getName())
					.setPrecision(precision)
					.setZeroThreshold(zeroThreshold);

				data.put(verificationField, getColumn(transformedDataset, predictionColumn));
			}

			for(String probabilityColumn : probabilityColumns){
				List<Feature> features = encoder.getFeatures(probabilityColumn);

				for(int i = 0; i < features.size(); i++){
					Feature feature = features.get(i);

					VerificationField verificationField = ModelUtil.createVerificationField(feature.getName())
						.setPrecision(precision)
						.setZeroThreshold(zeroThreshold);

					data.put(verificationField, getVectorColumn(transformedDataset, probabilityColumn, i));
				}
			}

			model.setModelVerification(ModelUtil.createModelVerification(data));
		}

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

	public PMMLBuilder extendSchema(Set<String> names){
		StructType schema = getSchema();
		PipelineModel pipelineModel = getPipelineModel();

		StructType transformedSchema = pipelineModel.transformSchema(schema);

		for(String name : names){
			StructField field = transformedSchema.apply(name);

			schema = schema.add(field);
		}

		setSchema(schema);

		return this;
	}

	public PMMLBuilder putOption(String key, Object value){
		return putOptions(Collections.singletonMap(key, value));
	}

	public PMMLBuilder putOptions(Map<String, ?> map){
		return putOptions(Pattern.compile(".*"), map);
	}

	public PMMLBuilder putOption(PipelineStage pipelineStage, String key, Object value){
		return putOptions(pipelineStage, Collections.singletonMap(key, value));
	}

	public PMMLBuilder putOptions(PipelineStage pipelineStage, Map<String, ?> map){
		return putOptions(Pattern.compile(pipelineStage.uid(), Pattern.LITERAL), map);
	}

	public PMMLBuilder putOptions(Pattern pattern, Map<String, ?> map){
		Map<RegexKey, Map<String, Object>> options = getOptions();

		RegexKey key = new RegexKey(pattern);

		Map<String, Object> patternOptions = options.get(key);
		if(patternOptions == null){
			patternOptions = new LinkedHashMap<>();

			options.put(key, patternOptions);
		}

		patternOptions.putAll(map);

		return this;
	}

	public PMMLBuilder verify(Dataset<Row> dataset){
		return verify(dataset, 1e-14, 1e-14);
	}

	public PMMLBuilder verify(Dataset<Row> dataset, double precision, double zeroThreshold){
		PipelineModel pipelineModel = getPipelineModel();

		Dataset<Row> transformedDataset = pipelineModel.transform(dataset);

		Verification verification = new Verification(dataset, transformedDataset)
			.setPrecision(precision)
			.setZeroThreshold(zeroThreshold);

		return setVerification(verification);
	}

	public StructType getSchema(){
		return this.schema;
	}

	public PMMLBuilder setSchema(StructType schema){
		this.schema = Objects.requireNonNull(schema);

		return this;
	}

	public PipelineModel getPipelineModel(){
		return this.pipelineModel;
	}

	public PMMLBuilder setPipelineModel(PipelineModel pipelineModel){
		this.pipelineModel = Objects.requireNonNull(pipelineModel);

		return this;
	}

	public Map<RegexKey, Map<String, Object>> getOptions(){
		return this.options;
	}

	private PMMLBuilder setOptions(Map<RegexKey, Map<String, Object>> options){
		this.options = Objects.requireNonNull(options);

		return this;
	}

	public Verification getVerification(){
		return this.verification;
	}

	private PMMLBuilder setVerification(Verification verification){
		this.verification = verification;

		return this;
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

	static
	private List<?> getColumn(Dataset<Row> dataset, String name){
		List<Row> rows = dataset.select(name)
			.collectAsList();

		return rows.stream()
			.map(row -> row.apply(0))
			.collect(Collectors.toList());
	}

	static
	private List<?> getVectorColumn(Dataset<Row> dataset, String name, int index){
		List<?> column = getColumn(dataset, name);

		return column.stream()
			.map(Vector.class::cast)
			.map(vector -> vector.apply(index))
			.collect(Collectors.toList());
	}

	static
	private void init(){
		ConverterFactory.checkVersion();
		ConverterFactory.checkApplicationClasspath();
		ConverterFactory.checkNoShading();
	}

	static
	public class Verification {

		private Dataset<Row> dataset = null;

		private Dataset<Row> transformedDataset = null;

		public Double precision = null;

		public Double zeroThreshold = null;


		private Verification(Dataset<Row> dataset, Dataset<Row> transformedDataset){
			setDataset(dataset);
			setTransformedDataset(transformedDataset);
		}

		public Dataset<Row> getDataset(){
			return this.dataset;
		}

		private Verification setDataset(Dataset<Row> dataset){
			this.dataset = dataset;

			return this;
		}

		public Dataset<Row> getTransformedDataset(){
			return this.transformedDataset;
		}

		private Verification setTransformedDataset(Dataset<Row> transformedDataset){
			this.transformedDataset = transformedDataset;

			return this;
		}

		public Double getPrecision(){
			return this.precision;
		}

		public Verification setPrecision(Double precision){
			this.precision = precision;

			return this;
		}

		public Double getZeroThreshold(){
			return this.zeroThreshold;
		}

		public Verification setZeroThreshold(Double zeroThreshold){
			this.zeroThreshold = zeroThreshold;

			return this;
		}
	}

	static {
		init();
	}
}