/*
 * Copyright (c) 2025 Villu Ruusmann
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
package org.jpmml.sparkml.feature;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.io.MoreFiles;
import com.google.common.io.RecursiveDeleteOption;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.util.MLReader;
import org.apache.spark.ml.util.MLWritable;
import org.apache.spark.ml.util.MLWriter;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.jpmml.sparkml.SparkMLTest;

import static org.junit.jupiter.api.Assertions.assertEquals;

abstract
public class DomainTest extends SparkMLTest {

	static
	protected void checkDataset(Map<String, List<Object>> expectedColumns, Dataset<Row> actualDs){
		Set<String> keys = expectedColumns.keySet();

		for(String key : keys){
			List<Object> expectedColumn = expectedColumns.get(key);

			List<Row> actualColumnRows = actualDs
				.select(key)
				.collectAsList();

			List<Object> actualColumn = actualColumnRows.stream()
				.map(row -> row.get(0))
				.collect(Collectors.toList());

			assertEquals(expectedColumn, actualColumn);
		}
	}

	static
	protected <S extends PipelineStage & MLWritable> S sparkClone(S stage) throws IOException {
		File tmpDir = createTempDir(stage);

		try {
			String path = tmpDir.getAbsolutePath();

			MLWriter writer = stage.write();

			writer
				.overwrite()
				.save(path);

			Class<?> stageClazz = stage.getClass();

			// The read method of the companion object
			Method readMethod = stageClazz.getDeclaredMethod("read");

			@SuppressWarnings("unchecked")
			MLReader<S> reader = (MLReader<S>)readMethod.invoke(null);

			return reader.load(path);
		} catch(ReflectiveOperationException roe){
			throw new RuntimeException(roe);
		} finally {
			MoreFiles.deleteRecursively(tmpDir.toPath(), RecursiveDeleteOption.ALLOW_INSECURE);
		}
	}

	static
	private File createTempDir(PipelineStage stage) throws IOException {
		File tmpFile = File.createTempFile("jpmml-sparkml-" + stage.uid(), "");

		if(!tmpFile.delete() || !tmpFile.mkdirs()){
			throw new IOException();
		}

		return tmpFile;
	}
}