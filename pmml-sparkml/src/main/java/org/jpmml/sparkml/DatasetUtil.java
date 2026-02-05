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

import java.io.File;
import java.io.FileFilter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.Reader;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.io.MoreFiles;
import com.google.common.io.RecursiveDeleteOption;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.DataFrameWriter;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalog.Catalog;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.execution.QueryExecution;
import org.apache.spark.sql.types.AtomicType;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.FractionalType;
import org.apache.spark.sql.types.IntegralType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.UserDefinedType;
import org.dmg.pmml.DataType;

public class DatasetUtil {

	private DatasetUtil(){
	}

	static
	public StructType loadSchema(File file) throws IOException {

		try(Reader reader = new FileReader(file, StandardCharsets.UTF_8)){
			StringWriter writer = new StringWriter();

			reader.transferTo(writer);

			String json = writer.toString();

			return (StructType)StructType.fromJson(json);
		}
	}

	static
	public void storeSchema(Dataset<Row> dataset, File file) throws IOException {
		storeSchema(dataset.schema(), file);
	}

	static
	public void storeSchema(StructType schema, File file) throws IOException {

		try(OutputStream os = new FileOutputStream(file)){
			String string = schema.json();

			os.write(string.getBytes("UTF-8"));
		}
	}


	static
	public Dataset<Row> loadCsv(SparkSession sparkSession, File file){
		return loadCsv(sparkSession, null, file);
	}

	static
	public Dataset<Row> loadCsv(SparkSession sparkSession, StructType schema, File file){
		DataFrameReader reader = sparkSession.read()
			.format("csv")
			.option("header", true)
			.option("nullValue", "N/A")
			.option("nanValue", "N/A");

		if(schema != null){
			reader = reader
				.schema(schema);
		} else

		{
			reader = reader
				.option("inferSchema", true);
		}

		return reader.load(file.getAbsolutePath());
	}

	static
	public void storeCsv(Dataset<Row> dataset, File file) throws IOException {
		File tmpDir = File.createTempFile("Dataset", "");
		if(!tmpDir.delete()){
			throw new IOException();
		}

		dataset = dataset.coalesce(1);

		DataFrameWriter<Row> writer = dataset.write()
			.format("csv")
			.option("header", "true")
			.option("nullValue", "N/A");

		writer.save(tmpDir.getAbsolutePath());

		FileFilter csvFileFilter = new FileFilter(){

			@Override
			public boolean accept(File file){
				String name = file.getName();

				return name.endsWith(".csv");
			}
		};

		File[] csvFiles = tmpDir.listFiles(csvFileFilter);
		if(csvFiles.length != 1){
			throw new IOException();
		}

		Files.copy(csvFiles[0].toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING);

		MoreFiles.deleteRecursively(tmpDir.toPath(), RecursiveDeleteOption.ALLOW_INSECURE);
	}

	static
	public Dataset<Row> toLibSVM(Dataset<Row> dataset){
		StructType schema = dataset.schema();

		StructField[] fields = schema.fields();

		// The last column
		String labelCol = fields[fields.length - 1].name();

		StringIndexer labelIndexer = new StringIndexer()
			.setInputCol(labelCol)
			.setOutputCol("label");

		StringIndexerModel labelIndexerModel = labelIndexer.fit(dataset);

		Dataset<Row> result = labelIndexerModel.transform(dataset);

		// All columns except for the last column
		List<String> featureCols = new ArrayList<>();
		for(int i = 0; i < fields.length - 1; i++){
			String featureCol = fields[i].name();

			featureCols.add(featureCol);
		}

		VectorAssembler vectorAssembler = new VectorAssembler()
			.setInputCols(featureCols.toArray(new String[featureCols.size()]))
			.setOutputCol("features");

		result = vectorAssembler.transform(result);

		result = result
			.select("label", "features");

		return result;
	}

	static
	public Dataset<Row> castColumn(Dataset<Row> dataset, String name, org.apache.spark.sql.types.DataType sparkDataType){
		Column column = dataset.apply(name).cast(sparkDataType);

		String tmpName = "tmp_" + name;

		return dataset.withColumn(tmpName, column).drop(name).withColumnRenamed(tmpName, name);
	}

	static
	public Dataset<Row> castColumns(Dataset<Row> dataset, StructType schema){
		StructType prevSchema = dataset.schema();

		StructField[] fields = schema.fields();
		for(StructField field : fields){
			StructField prevField;

			try {
				prevField = prevSchema.apply(field.name());
			} catch(IllegalArgumentException iae){
				continue;
			}

			if(!Objects.equals(field.dataType(), prevField.dataType())){
				dataset = castColumn(dataset, field.name(), field.dataType());
			}
		}

		return dataset;
	}

	static
	public LogicalPlan createAnalyzedLogicalPlan(SparkSession sparkSession, StructType schema, String statement){
		String tableName = "sql2pmml_" + DatasetUtil.ID.getAndIncrement();

		statement = statement.replace("__THIS__", tableName);

		Dataset<Row> dataset = sparkSession.createDataFrame(Collections.emptyList(), schema);

		dataset.createOrReplaceTempView(tableName);

		try {
			QueryExecution queryExecution = sparkSession.sql(statement).queryExecution();

			return queryExecution.analyzed();
		} finally {
			Catalog catalog = sparkSession.catalog();

			catalog.dropTempView(tableName);
		}
	}

	static
	public DataType translateDataType(org.apache.spark.sql.types.DataType sparkDataType){

		if(sparkDataType instanceof AtomicType){
			return translateAtomicType((AtomicType)sparkDataType);
		} else

		if(sparkDataType instanceof UserDefinedType){
			return translateUserDefinedType((UserDefinedType)sparkDataType);
		} else

		{
			throw new SparkMLException("Expected atomic data type, got " + sparkDataType.typeName());
		}
	}

	static
	public DataType translateAtomicType(org.apache.spark.sql.types.AtomicType atomicType){

		if(atomicType instanceof StringType){
			return DataType.STRING;
		} else

		if(atomicType instanceof IntegralType){
			return translateIntegralType((IntegralType)atomicType);
		} else

		if(atomicType instanceof FractionalType){
			return translateFractionalType((FractionalType)atomicType);
		} else

		if(atomicType instanceof BooleanType){
			return DataType.BOOLEAN;
		} else

		{
			throw new SparkMLException("Expected string, integral, fractional or boolean data type, got " + atomicType.typeName());
		}
	}

	static
	public DataType translateIntegralType(IntegralType integralType){
		return DataType.INTEGER;
	}

	static
	public DataType translateFractionalType(FractionalType fractionalType){

		if(fractionalType instanceof FloatType){
			return DataType.FLOAT;
		} else

		if(fractionalType instanceof DoubleType){
			return DataType.DOUBLE;
		} else

		{
			throw new SparkMLException("Expected float or double data type, got " + fractionalType.typeName());
		}
	}

	static
	public DataType translateUserDefinedType(UserDefinedType userDefinedType){

		if(userDefinedType instanceof VectorUDT){
			return DataType.DOUBLE;
		} else

		{
			throw new SparkMLException("Expected vector data type, got " + userDefinedType.typeName());
		}
	}

	private static final AtomicInteger ID = new AtomicInteger(1);
}