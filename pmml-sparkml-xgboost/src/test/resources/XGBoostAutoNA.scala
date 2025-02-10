import java.io.File

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.types.FloatType
import org.jpmml.sparkml.{ArchiveUtil, DatasetUtil, PipelineModelUtil}
import org.jpmml.sparkml.feature.InvalidCategoryTransformer

var df = DatasetUtil.loadCsv(spark, new File("csv/AutoNA.csv"))

DatasetUtil.storeSchema(df, new File("schema/AutoNA.json"))

val cat_cols = Array("cylinders", "model_year", "origin")
val cont_cols = Array("acceleration", "displacement", "horsepower", "weight")

val indexer = new StringIndexer().setInputCols(cat_cols).setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col)).setHandleInvalid("keep")
val indexTransformer = new InvalidCategoryTransformer().setInputCols(indexer.getOutputCols).setOutputCols(cat_cols.map(cat_col => "idxTransformed_" + cat_col))

val assembler = new VectorAssembler().setInputCols(indexTransformer.getOutputCols ++ cont_cols).setOutputCol("featureVector").setHandleInvalid("keep")

val regressor = new XGBoostRegressor(Map("objective" -> "reg:squarederror", "num_round" -> 101, "num_workers" -> 1, "tree_method" -> "hist")).setLabelCol("mpg").setFeaturesCol(assembler.getOutputCol).setFeatureTypes(Array("c", "c", "c", "q", "q", "q", "q"))

val pipeline = new Pipeline().setStages(Array(indexer, indexTransformer, assembler, regressor))
val pipelineModel = pipeline.fit(df)

PipelineModelUtil.storeZip(pipelineModel, new File("pipeline/XGBoostAutoNA.zip"))

var xgbDf = pipelineModel.transform(df)
xgbDf = xgbDf.selectExpr("prediction as mpg")
xgbDf = DatasetUtil.castColumn(xgbDf, "mpg", FloatType)

DatasetUtil.storeCsv(xgbDf, new File("csv/XGBoostAutoNA.csv"))
