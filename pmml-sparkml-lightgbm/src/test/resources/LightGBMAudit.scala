import java.nio.file.{Files, Paths}

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.jpmml.sparkml.{DatasetUtil, PipelineModelUtil}

var df = spark.read.option("header", "true").option("inferSchema", "true").csv("csv/Audit.csv")
df = DatasetUtil.castColumn(df, "Adjusted", StringType)

val cat_cols = Array("Education", "Employment", "Gender", "Marital", "Occupation")
val cont_cols = Array("Age", "Hours", "Income")

val labelIndexer = new StringIndexer().setInputCol("Adjusted").setOutputCol("idx_Adjusted")

val indexer = new StringIndexer().setInputCols(cat_cols).setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col))
val assembler = new VectorAssembler().setInputCols(indexer.getOutputCols ++ cont_cols).setOutputCol("featureVector")

val classifier = new LightGBMClassifier().setNumIterations(101).setLabelCol(labelIndexer.getOutputCol).setFeaturesCol(assembler.getOutputCol)

val pipeline = new Pipeline().setStages(Array(labelIndexer, indexer, assembler, classifier))
val pipelineModel = pipeline.fit(df)

PipelineModelUtil.storeZip(pipelineModel, "pipelines/LightGBMAudit.zip")

val predLabel = udf{ (value: Float) => value.toInt.toString }
val vectorToColumn = udf{ (vec: Vector, index: Int) => vec(index) }

var lgbDf = pipelineModel.transform(df)
lgbDf = lgbDf.selectExpr("prediction", "probability")
lgbDf = lgbDf.withColumn("Adjusted", predLabel(lgbDf("prediction"))).drop("prediction")
lgbDf = lgbDf.withColumn("probability(0)", vectorToColumn(lgbDf("probability"), lit(0))).withColumn("probability(1)", vectorToColumn(lgbDf("probability"), lit(1))).drop("probability").drop("probability")

DatasetUtil.storeCsv(lgbDf, "csv/LightGBMAudit.csv")
