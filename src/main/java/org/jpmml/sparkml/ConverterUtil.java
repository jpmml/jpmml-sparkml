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
package org.jpmml.sparkml;

import java.lang.reflect.Constructor;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.Binarizer;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.ml.feature.ChiSqSelectorModel;
import org.apache.spark.ml.feature.ColumnPruner;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.RFormulaModel;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorAttributeRewriter;
import org.apache.spark.ml.feature.VectorSlicer;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.feature.BinarizerConverter;
import org.jpmml.sparkml.feature.BucketizerConverter;
import org.jpmml.sparkml.feature.ChiSqSelectorModelConverter;
import org.jpmml.sparkml.feature.ColumnPrunerConverter;
import org.jpmml.sparkml.feature.MinMaxScalerModelConverter;
import org.jpmml.sparkml.feature.OneHotEncoderConverter;
import org.jpmml.sparkml.feature.PCAModelConverter;
import org.jpmml.sparkml.feature.RFormulaModelConverter;
import org.jpmml.sparkml.feature.StandardScalerModelConverter;
import org.jpmml.sparkml.feature.StringIndexerModelConverter;
import org.jpmml.sparkml.feature.VectorAssemblerConverter;
import org.jpmml.sparkml.feature.VectorAttributeRewriterConverter;
import org.jpmml.sparkml.feature.VectorSlicerConverter;
import org.jpmml.sparkml.model.DecisionTreeClassificationModelConverter;
import org.jpmml.sparkml.model.DecisionTreeRegressionModelConverter;
import org.jpmml.sparkml.model.GBTClassificationModelConverter;
import org.jpmml.sparkml.model.GBTRegressionModelConverter;
import org.jpmml.sparkml.model.KMeansModelConverter;
import org.jpmml.sparkml.model.LinearRegressionModelConverter;
import org.jpmml.sparkml.model.LogisticRegressionModelConverter;
import org.jpmml.sparkml.model.RandomForestClassificationModelConverter;
import org.jpmml.sparkml.model.RandomForestRegressionModelConverter;

public class ConverterUtil {

	private ConverterUtil(){
	}

	static
	public PMML toPMML(StructType schema, PipelineModel pipelineModel){
		FeatureMapper featureMapper = new FeatureMapper(schema);

		Transformer[] stages = pipelineModel.stages();
		for(Transformer stage : stages){
			TransformerConverter<?> converter = ConverterUtil.createConverter(stage);

			if(converter instanceof FeatureConverter){
				FeatureConverter<?> featureConverter = (FeatureConverter<?>)converter;

				featureMapper.append(featureConverter);
			} else

			if(converter instanceof ModelConverter){
				ModelConverter<?> modelConverter = (ModelConverter<?>)converter;

				Schema featureSchema = featureMapper.createSchema((Model<?>)stage);

				org.dmg.pmml.Model model = modelConverter.encodeModel(featureSchema);

				PMML pmml = featureMapper.encodePMML(model)
					.setHeader(PMMLUtil.createHeader("JPMML-SparkML", "1.0-SNAPSHOT"));

				return pmml;
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		throw new IllegalArgumentException();
	}

	static
	public FeatureConverter<?> createFeatureConverter(Transformer transformer){
		return (FeatureConverter<?>)createConverter(transformer);
	}

	static
	public ModelConverter<?> createModelConverter(Transformer transformer){
		return (ModelConverter<?>)createConverter(transformer);
	}

	static
	public <T extends Transformer> TransformerConverter<T> createConverter(T transformer){
		Class<? extends Transformer> clazz = transformer.getClass();

		Class<? extends TransformerConverter> converterClazz = getConverterClazz(clazz);
		if(converterClazz == null){
			throw new IllegalArgumentException("Transformer class " + clazz.getName() + " is not supported");
		}

		try {
			Constructor<?> constructor = converterClazz.getDeclaredConstructor(clazz);

			return (TransformerConverter)constructor.newInstance(transformer);
		} catch(Exception e){
			throw new IllegalArgumentException(e);
		}
	}

	static
	public Class<? extends TransformerConverter> getConverterClazz(Class<? extends Transformer> clazz){
		return ConverterUtil.converters.get(clazz);
	}

	static
	public void putConverterClazz(Class<? extends Transformer> clazz, Class<? extends TransformerConverter<?>> converterClazz){
		ConverterUtil.converters.put(clazz, converterClazz);
	}

	private static final Map<Class<? extends Transformer>, Class<? extends TransformerConverter>> converters = new LinkedHashMap<>();

	static {
		// Features
		converters.put(Binarizer.class, BinarizerConverter.class);
		converters.put(Bucketizer.class, BucketizerConverter.class);
		converters.put(ChiSqSelectorModel.class, ChiSqSelectorModelConverter.class);
		converters.put(ColumnPruner.class, ColumnPrunerConverter.class);
		converters.put(MinMaxScalerModel.class, MinMaxScalerModelConverter.class);
		converters.put(OneHotEncoder.class, OneHotEncoderConverter.class);
		converters.put(PCAModel.class, PCAModelConverter.class);
		converters.put(RFormulaModel.class, RFormulaModelConverter.class);
		converters.put(StandardScalerModel.class, StandardScalerModelConverter.class);
		converters.put(StringIndexerModel.class, StringIndexerModelConverter.class);
		converters.put(VectorAssembler.class, VectorAssemblerConverter.class);
		converters.put(VectorAttributeRewriter.class, VectorAttributeRewriterConverter.class);
		converters.put(VectorSlicer.class, VectorSlicerConverter.class);

		// Models
		converters.put(DecisionTreeClassificationModel.class, DecisionTreeClassificationModelConverter.class);
		converters.put(DecisionTreeRegressionModel.class, DecisionTreeRegressionModelConverter.class);
		converters.put(GBTClassificationModel.class, GBTClassificationModelConverter.class);
		converters.put(GBTRegressionModel.class, GBTRegressionModelConverter.class);
		converters.put(KMeansModel.class, KMeansModelConverter.class);
		converters.put(LinearRegressionModel.class, LinearRegressionModelConverter.class);
		converters.put(LogisticRegressionModel.class, LogisticRegressionModelConverter.class);
		converters.put(RandomForestClassificationModel.class, RandomForestClassificationModelConverter.class);
		converters.put(RandomForestRegressionModel.class, RandomForestRegressionModelConverter.class);
	}
}