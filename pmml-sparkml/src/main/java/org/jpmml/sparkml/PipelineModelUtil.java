/*
 * Copyright (c) 2017 Villu Ruusmann
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

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.google.common.io.MoreFiles;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.util.MLReader;
import org.apache.spark.ml.util.MLWriter;
import org.apache.spark.sql.SparkSession;
import org.jpmml.model.ReflectionUtil;

public class PipelineModelUtil {

	private PipelineModelUtil(){
	}

	static
	public void addStage(PipelineModel pipelineModel, int index, Transformer transformer){
		List<Transformer> stages = new ArrayList<>(Arrays.asList(pipelineModel.stages()));

		stages.add(index, transformer);

		ReflectionUtil.setFieldValue(PipelineModelUtil.FIELD_STAGES, pipelineModel, stages.toArray(new Transformer[stages.size()]));
	}

	static
	public Transformer removeStage(PipelineModel pipelineModel, int index){
		List<Transformer> stages = new ArrayList<>(Arrays.asList(pipelineModel.stages()));

		Transformer result = stages.remove(index);

		ReflectionUtil.setFieldValue(PipelineModelUtil.FIELD_STAGES, pipelineModel, stages.toArray(new Transformer[stages.size()]));

		return result;
	}

	static
	public PipelineModel load(SparkSession sparkSession, File dir) throws IOException {
		MLReader<PipelineModel> mlReader = PipelineModel.read();
		mlReader.session(sparkSession);

		return mlReader.load(dir.getAbsolutePath());
	}

	static
	public PipelineModel loadZip(SparkSession sparkSession, File file) throws IOException {
		File tmpDir = ArchiveUtil.uncompress(file);

		PipelineModel pipelineModel = load(sparkSession, tmpDir);

		MoreFiles.deleteRecursively(tmpDir.toPath());

		return pipelineModel;
	}

	static
	public void store(PipelineModel pipelineModel, File dir) throws IOException {
		MLWriter mlWriter = new PipelineModel.PipelineModelWriter(pipelineModel);

		mlWriter.save(dir.getAbsolutePath());
	}

	static
	public void storeZip(PipelineModel pipelineModel, File file) throws IOException {
		File tmpDir = File.createTempFile("PipelineModel", "");
		if(!tmpDir.delete()){
			throw new IOException();
		}

		store(pipelineModel, tmpDir);

		ArchiveUtil.compress(tmpDir, file);

		MoreFiles.deleteRecursively(tmpDir.toPath());
	}

	private static final Field FIELD_STAGES = ReflectionUtil.getField(PipelineModel.class, "stages");
}