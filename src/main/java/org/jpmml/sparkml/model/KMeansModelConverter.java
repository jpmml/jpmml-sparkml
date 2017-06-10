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
import java.util.Collections;
import java.util.List;

import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.dmg.pmml.CompareFunction;
import org.dmg.pmml.ComparisonMeasure;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.SquaredEuclidean;
import org.dmg.pmml.clustering.Cluster;
import org.dmg.pmml.clustering.ClusteringModel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.PMMLUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.clustering.ClusteringModelUtil;
import org.jpmml.sparkml.ClusteringModelConverter;
import org.jpmml.sparkml.VectorUtil;

public class KMeansModelConverter extends ClusteringModelConverter<KMeansModel> {

	public KMeansModelConverter(KMeansModel model){
		super(model);
	}

	@Override
	public ClusteringModel encodeModel(Schema schema){
		KMeansModel model = getTransformer();

		List<Feature> features = schema.getFeatures();

		List<Cluster> clusters = new ArrayList<>();

		Vector[] clusterCenters = model.clusterCenters();
		for(int i = 0; i < clusterCenters.length; i++){
			Cluster cluster = new Cluster()
				.setId(String.valueOf(i))
				.setArray(PMMLUtil.createRealArray(VectorUtil.toList(clusterCenters[i])));

			clusters.add(cluster);
		}

		ComparisonMeasure comparisonMeasure = new ComparisonMeasure(ComparisonMeasure.Kind.DISTANCE)
			.setCompareFunction(CompareFunction.ABS_DIFF)
			.setMeasure(new SquaredEuclidean());

		ClusteringModel clusteringModel = new ClusteringModel(MiningFunction.CLUSTERING, ClusteringModel.ModelClass.CENTER_BASED, clusters.size(), ModelUtil.createMiningSchema(schema), comparisonMeasure, ClusteringModelUtil.createClusteringFields(features), clusters)
			.setOutput(ClusteringModelUtil.createOutput(FieldName.create("cluster"), Collections.<Cluster>emptyList()));

		return clusteringModel;
	}
}