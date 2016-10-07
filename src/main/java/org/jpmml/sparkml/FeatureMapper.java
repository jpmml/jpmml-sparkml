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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.param.shared.HasFeaturesCol;
import org.apache.spark.ml.param.shared.HasLabelCol;
import org.apache.spark.ml.param.shared.HasOutputCol;
import org.apache.spark.ml.param.shared.HasPredictionCol;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.IntegralType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.Value;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ListFeature;
import org.jpmml.converter.PMMLMapper;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.WildcardFeature;

public class FeatureMapper extends PMMLMapper {

	private StructType schema = null;

	private Map<String, List<Feature>> columnFeatures = new LinkedHashMap<>();


	public FeatureMapper(StructType schema){
		this.schema = schema;
	}

	public void append(FeatureConverter<?> featureConverter){
		Transformer transformer = featureConverter.getTransformer();

		List<Feature> features = featureConverter.encodeFeatures(this);

		if(transformer instanceof HasOutputCol){
			HasOutputCol hasOutputCol = (HasOutputCol)transformer;

			String outputCol = hasOutputCol.getOutputCol();

			putFeatures(outputCol, features);
		}
	}

	public void append(ModelConverter<?> modelConverter){
		Model<?> model = modelConverter.getTransformer();

		List<Feature> features = modelConverter.encodeFeatures(this);

		if(model instanceof HasPredictionCol){
			HasPredictionCol hasPredictionCol = (HasPredictionCol)model;

			String predictionCol = hasPredictionCol.getPredictionCol();

			putFeatures(predictionCol, features);
		}
	}

	public Schema createSchema(ModelConverter<?> modelConverter){
		FieldName targetField;
		List<String> targetCategories = null;

		Model<?> model = modelConverter.getTransformer();

		if(model instanceof PredictionModel){
			HasLabelCol hasLabelCol = (HasLabelCol)model;

			Feature feature = getOnlyFeature(hasLabelCol.getLabelCol());

			targetField = feature.getName();

			MiningFunction miningFunction = modelConverter.getMiningFunction();
			switch(miningFunction){
				case CLASSIFICATION:
					{
						if(feature instanceof ListFeature){
							ListFeature listFeature = (ListFeature)feature;

							targetCategories = listFeature.getValues();
						} else

						{
							ContinuousFeature continuousFeature = (ContinuousFeature)feature;

							// XXX
							targetCategories = Arrays.asList("0", "1");

							DataField dataField = toCategorical(targetField, targetCategories);

							ListFeature listFeature = new ListFeature(dataField, targetCategories);

							this.columnFeatures.put(hasLabelCol.getLabelCol(), Collections.<Feature>singletonList(listFeature));
						}
					}
					break;
				case REGRESSION:
					{
						DataField dataField = toContinuous(targetField);

						dataField.setDataType(DataType.DOUBLE);
					}
					break;
				default:
					break;
			}
		} else

		if(model instanceof KMeansModel){
			targetField = null;
		} else

		{
			throw new IllegalArgumentException();
		}

		Map<FieldName, DataField> dataFields = getDataFields();

		List<FieldName> activeFields = new ArrayList<>(dataFields.keySet());

		if(targetField != null){
			activeFields.remove(targetField);
		}

		HasFeaturesCol hasFeaturesCol = (HasFeaturesCol)model;

		List<Feature> features = getFeatures(hasFeaturesCol.getFeaturesCol());

		if(model instanceof PredictionModel){
			PredictionModel<?, ?> predictionModel = (PredictionModel<?, ?>)model;

			int numFeatures = predictionModel.numFeatures();
			if(numFeatures != -1 && features.size() != numFeatures){
				throw new IllegalArgumentException("Expected " + numFeatures + " features, got " + features.size() + " features");
			}
		}

		Schema result = new Schema(targetField, targetCategories, activeFields, features);

		return result;
	}

	public DataField toContinuous(FieldName name){
		DataField dataField = getDataField(name);

		if(dataField == null){
			throw new IllegalArgumentException();
		}

		dataField.setOpType(OpType.CONTINUOUS);

		return dataField;
	}

	public DataField toCategorical(FieldName name, List<String> categories){
		DataField dataField = getDataField(name);

		if(dataField == null){
			throw new IllegalArgumentException();
		}

		dataField.setOpType(OpType.CATEGORICAL);

		List<Value> values = dataField.getValues();
		if(values.size() > 0){
			throw new IllegalArgumentException();
		}

		values.addAll(PMMLUtil.createValues(categories));

		return dataField;
	}

	public boolean hasFeatures(String column){
		return this.columnFeatures.containsKey(column);
	}

	public Feature getOnlyFeature(String column){
		List<Feature> features = getFeatures(column);

		return Iterables.getOnlyElement(features);
	}

	public List<Feature> getFeatures(String column){
		List<Feature> features = this.columnFeatures.get(column);

		if(features == null){
			FieldName name = FieldName.create(column);

			DataField dataField = getDataField(name);
			if(dataField == null){
				dataField = createDataField(name);
			}

			Feature feature;

			DataType dataType = dataField.getDataType();
			switch(dataType){
				case STRING:
					feature = new WildcardFeature(dataField);
					break;
				case INTEGER:
				case DOUBLE:
					feature = new ContinuousFeature(dataField);
					break;
				case BOOLEAN:
					feature = new BooleanFeature(dataField);
					break;
				default:
					throw new IllegalArgumentException();
			}

			return Collections.singletonList(feature);
		}

		return features;
	}

	public List<Feature> getFeatures(String column, int[] indices){
		List<Feature> features = getFeatures(column);

		List<Feature> result = new ArrayList<>();

		for(int i = 0; i < indices.length; i++){
			int index = indices[i];

			Feature feature = features.get(index);

			result.add(feature);
		}

		return result;
	}

	public void putFeatures(String column, List<Feature> features){
		checkColumn(column);

		this.columnFeatures.put(column, features);
	}

	public DataField createDataField(FieldName name){
		StructField field = this.schema.apply(name.getValue());

		org.apache.spark.sql.types.DataType sparkDataType = field.dataType();

		if(sparkDataType instanceof StringType){
			return createDataField(name, OpType.CATEGORICAL, DataType.STRING);
		} else

		if(sparkDataType instanceof IntegralType){
			return createDataField(name, OpType.CONTINUOUS, DataType.INTEGER);
		} else

		if(sparkDataType instanceof DoubleType){
			return createDataField(name, OpType.CONTINUOUS, DataType.DOUBLE);
		} else

		if(sparkDataType instanceof BooleanType){
			return createDataField(name, OpType.CATEGORICAL, DataType.BOOLEAN);
		} else

		{
			throw new IllegalArgumentException("Expected string, integral, double or boolean type, got " + sparkDataType.typeName() + " type");
		}
	}

	public void removeDataField(FieldName name){
		Map<FieldName, DataField> dataFields = getDataFields();

		DataField dataField = dataFields.remove(name);
		if(dataField == null){
			throw new IllegalArgumentException();
		}
	}

	private void checkColumn(String column){
		List<Feature> features = this.columnFeatures.get(column);

		if(features != null && features.size() > 0){
			Feature feature = Iterables.getOnlyElement(features);

			if(!(feature instanceof WildcardFeature)){
				throw new IllegalArgumentException(column);
			}
		}
	}
}