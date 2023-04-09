import java.nio.file.{Files, Paths}

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.jpmml.sparkml.ZipUtil

var df = spark.read.option("header", "true").option("inferSchema", "true").csv("csv/Audit.csv")
df = df.withColumn("AdjustedTmp", df("Adjusted").cast(StringType)).drop("Adjusted").withColumnRenamed("AdjustedTmp", "Adjusted")

val cat_cols = Array("Education", "Employment", "Gender", "Marital", "Occupation")
val cont_cols = Array("Age", "Hours", "Income")

val labelIndexer = new StringIndexer().setInputCol("Adjusted").setOutputCol("idx_Adjusted")

val indexer = new StringIndexer().setInputCols(cat_cols).setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col))
val assembler = new VectorAssembler().setInputCols(indexer.getOutputCols ++ cont_cols).setOutputCol("featureVector")

val classifier = new LightGBMClassifier().setNumIterations(101).setLabelCol(labelIndexer.getOutputCol).setFeaturesCol(assembler.getOutputCol)

val pipeline = new Pipeline().setStages(Array(labelIndexer, indexer, assembler, classifier))
val pipelineModel = pipeline.fit(df)

val tmpDir = Files.createTempDirectory("_jpmml_lightgbm_").toFile()
pipelineModel.write.overwrite().save(tmpDir.getAbsolutePath())
ZipUtil.compress(tmpDir, new File("pipelines/LightGBMAudt.zip"))

val vectorToColumn = udf{ (vec: Vector, index: Int) => vec(index) }

var lgbDf = pipelineModel.transform(df)
lgbDf = lgbDf.selectExpr("prediction as Adjusted", "probability")
lgbDf = lgbDf.withColumn("AdjustedTmp", lgbDf("Adjusted").cast(IntegerType).cast(StringType)).drop("Adjusted").withColumnRenamed("AdjustedTmp", "Adjusted")
lgbDf = lgbDf.withColumn("probability(0)", vectorToColumn(lgbDf("probability"), lit(0))).withColumn("probability(1)", vectorToColumn(lgbDf("probability"), lit(1))).drop("probability")

lgbDf.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("csv/LightGBMAudit")
