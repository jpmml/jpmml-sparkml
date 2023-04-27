import java.io.File

import ml.dmlc.xgboost4j.scala.spark.{TrackerConf, XGBoostRegressor}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.types.FloatType
import org.jpmml.sparkml.{DatasetUtil, PipelineModelUtil}

var df = DatasetUtil.loadCsv(spark, new File("csv/Housing.csv"))

DatasetUtil.storeSchema(df, new File("schema/Housing.json"))

val cat_cols = Array("CHAS", "RAD", "TAX")
val cont_cols = Array("CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "PTRATIO", "B", "LSTAT")

val assembler = new VectorAssembler().setInputCols(cat_cols ++ cont_cols).setOutputCol("featureVector")
val indexer = new VectorIndexer().setInputCol(assembler.getOutputCol).setOutputCol("catFeatureVector")

val trackerConf = TrackerConf(0, "scala")
val regressor = new XGBoostRegressor(Map("objective" -> "reg:squarederror", "num_round" -> 101, "num_workers" -> 1, "tracker_conf" -> trackerConf)).setMissing(-1).setLabelCol("MEDV").setFeaturesCol(indexer.getOutputCol)

val pipeline = new Pipeline().setStages(Array(assembler, indexer, regressor))
val pipelineModel = pipeline.fit(df)

PipelineModelUtil.storeZip(pipelineModel, new File("pipeline/XGBoostHousing.zip"))

var xgbDf = pipelineModel.transform(df)
xgbDf = xgbDf.selectExpr("prediction as MEDV")
xgbDf = DatasetUtil.castColumn(xgbDf, "MEDV", FloatType)

DatasetUtil.storeCsv(xgbDf, new File("csv/XGBoostHousing.csv"))
