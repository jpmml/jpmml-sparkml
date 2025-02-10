import java.io.File

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.StringType
import org.jpmml.sparkml.{DatasetUtil, PipelineModelUtil}
import org.jpmml.sparkml.feature.SparseToDenseTransformer

var df = DatasetUtil.loadCsv(spark, new File("csv/Audit.csv"))
df = DatasetUtil.castColumn(df, "Adjusted", StringType)

DatasetUtil.storeSchema(df, new File("schema/Audit.json"))

val cat_cols = Array("Education", "Employment", "Gender", "Marital", "Occupation")
val cont_cols = Array("Age", "Hours", "Income")

val labelIndexer = new StringIndexer().setInputCol("Adjusted").setOutputCol("idx_Adjusted")

val indexer = new StringIndexer().setInputCols(cat_cols).setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col))
val ohe = new OneHotEncoder().setHandleInvalid("keep").setDropLast(true).setInputCols(indexer.getOutputCols).setOutputCols(cat_cols.map(cat_col => "ohe_" + cat_col))
val assembler = new VectorAssembler().setInputCols(ohe.getOutputCols ++ cont_cols).setOutputCol("featureVector")

val sparse2dense = new SparseToDenseTransformer().setInputCol(assembler.getOutputCol).setOutputCol("denseFeatureVec")

val classifier = new XGBoostClassifier(Map("objective" -> "binary:logistic", "num_round" -> 101)).setLabelCol(labelIndexer.getOutputCol).setFeaturesCol(sparse2dense.getOutputCol)

val pipeline = new Pipeline().setStages(Array(labelIndexer, indexer, ohe, assembler, sparse2dense, classifier))
val pipelineModel = pipeline.fit(df)

PipelineModelUtil.storeZip(pipelineModel, new File("pipeline/XGBoostAudit.zip"))

val predLabel = udf{ (value: Float) => value.toInt.toString }
val vectorToColumn = udf{ (vec: Vector, index: Int) => vec(index).toFloat }

var xgbDf = pipelineModel.transform(df)
xgbDf = xgbDf.selectExpr("prediction", "probability")
xgbDf = xgbDf.withColumn("Adjusted", predLabel(xgbDf("prediction"))).drop("prediction")
xgbDf = xgbDf.withColumn("probability(0)", vectorToColumn(xgbDf("probability"), lit(0))).withColumn("probability(1)", vectorToColumn(xgbDf("probability"), lit(1))).drop("probability").drop("probability")

DatasetUtil.storeCsv(xgbDf, new File("csv/XGBoostAudit.csv"))
