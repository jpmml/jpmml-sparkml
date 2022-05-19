import java.nio.file.{Files, Paths}

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMRegressor
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.types.StringType
import org.jpmml.sparkml.PMMLBuilder

var df = spark.read.option("header", "true").option("inferSchema", "true").csv("csv/Auto.csv")
df = df.withColumn("originTmp", df("origin").cast(StringType)).drop("origin").withColumnRenamed("originTmp", "origin")

val cat_cols = Array("cylinders", "model_year", "origin")
val cont_cols = Array("acceleration", "displacement", "horsepower", "weight")

val indexer = new StringIndexer().setInputCols(cat_cols).setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col))
val assembler = new VectorAssembler().setInputCols(indexer.getOutputCols ++ cont_cols).setOutputCol("featureVector")

val regressor = new LightGBMRegressor().setNumIterations(101).setLabelCol("mpg").setFeaturesCol(assembler.getOutputCol)

val pipeline = new Pipeline().setStages(Array(indexer, assembler, regressor))
val pipelineModel = pipeline.fit(df)

var lgbDf = pipelineModel.transform(df)
lgbDf = lgbDf.selectExpr("prediction as mpg")
lgbDf.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("csv/LightGBMAuto")

//pipelineModel.save("pipeline/LightGBMAuto")

val pmmlBytes = new PMMLBuilder(df.schema, pipelineModel).buildByteArray()
Files.write(Paths.get("pmml/LightGBMAuto.pmml"), pmmlBytes)
