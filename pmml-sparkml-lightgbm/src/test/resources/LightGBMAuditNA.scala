import java.io.File

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.StringType
import org.jpmml.sparkml.{DatasetUtil, PipelineModelUtil}
import org.jpmml.sparkml.feature.InvalidCategoryTransformer

var df = DatasetUtil.loadCsv(spark, new File("csv/AuditNA.csv"))
df = DatasetUtil.castColumn(df, "Adjusted", StringType)

DatasetUtil.storeSchema(df, new File("schema/AuditNA.json"))

val cat_cols = Array("Education", "Employment", "Gender", "Marital", "Occupation")
val cont_cols = Array("Age", "Hours", "Income")

val labelIndexer = new StringIndexer().setInputCol("Adjusted").setOutputCol("idx_Adjusted")

val indexer = new StringIndexer().setInputCols(cat_cols).setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col)).setHandleInvalid("keep")
val indexTransformer = new InvalidCategoryTransformer().setInputCols(indexer.getOutputCols).setOutputCols(cat_cols.map(cat_col => "idxTransformed_" + cat_col))

val assembler = new VectorAssembler().setInputCols(indexTransformer.getOutputCols ++ cont_cols).setOutputCol("featureVector").setHandleInvalid("keep")

val classifier = new LightGBMClassifier().setObjective("binary").setNumIterations(101).setLabelCol(labelIndexer.getOutputCol).setFeaturesCol(assembler.getOutputCol)

val pipeline = new Pipeline().setStages(Array(labelIndexer, indexer, indexTransformer, assembler, classifier))
val pipelineModel = pipeline.fit(df)

PipelineModelUtil.storeZip(pipelineModel, new File("pipeline/LightGBMAuditNA.zip"))

val predLabel = udf{ (value: Float) => value.toInt.toString }
val vectorToColumn = udf{ (vec: Vector, index: Int) => vec(index) }

var lgbDf = pipelineModel.transform(df)
lgbDf = lgbDf.selectExpr("prediction", "probability")
lgbDf = lgbDf.withColumn("Adjusted", predLabel(lgbDf("prediction"))).drop("prediction")
lgbDf = lgbDf.withColumn("probability(0)", vectorToColumn(lgbDf("probability"), lit(0))).withColumn("probability(1)", vectorToColumn(lgbDf("probability"), lit(1))).drop("probability").drop("probability")

DatasetUtil.storeCsv(lgbDf, new File("csv/LightGBMAuditNA.csv"))
