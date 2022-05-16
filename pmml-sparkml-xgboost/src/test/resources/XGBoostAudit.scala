import java.nio.file.{Files, Paths}

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.jpmml.sparkml.PMMLBuilder
import org.jpmml.sparkml.xgboost.SparseToDenseTransformer

var df = spark.read.option("header", "true").option("inferSchema", "true").csv("csv/Audit.csv")
df = df.withColumn("AdjustedTmp", df("Adjusted").cast(StringType)).drop("Adjusted").withColumnRenamed("AdjustedTmp", "Adjusted")

val cat_cols = Array("Education", "Employment", "Gender", "Marital", "Occupation")
val cont_cols = Array("Age", "Hours", "Income")

val labelIndexer = new StringIndexer().setInputCol("Adjusted").setOutputCol("idx_Adjusted")

val indexer = new StringIndexer().setInputCols(cat_cols).setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col))
val ohe = new OneHotEncoder().setDropLast(false).setInputCols(indexer.getOutputCols).setOutputCols(cat_cols.map(cat_col => "ohe_" + cat_col))
val assembler = new VectorAssembler().setInputCols(ohe.getOutputCols ++ cont_cols).setOutputCol("featureVector")

val sparse2dense = new SparseToDenseTransformer().setInputCol(assembler.getOutputCol).setOutputCol("denseFeatureVec")

val classifier = new XGBoostClassifier(Map("objective" -> "binary:logistic", "num_round" -> 101)).setLabelCol(labelIndexer.getOutputCol).setFeaturesCol(sparse2dense.getOutputCol)

val pipeline = new Pipeline().setStages(Array(labelIndexer, indexer, ohe, assembler, sparse2dense, classifier))
val pipelineModel = pipeline.fit(df)

val vectorToColumn = udf{ (vec: Vector, index: Int) => vec(index).toFloat }

var xgbDf = pipelineModel.transform(df)
xgbDf = xgbDf.selectExpr("prediction as Adjusted", "probability")
xgbDf = xgbDf.withColumn("AdjustedTmp", xgbDf("Adjusted").cast(IntegerType).cast(StringType)).drop("Adjusted").withColumnRenamed("AdjustedTmp", "Adjusted")
xgbDf = xgbDf.withColumn("probability(0)", vectorToColumn(xgbDf("probability"), lit(0))).withColumn("probability(1)", vectorToColumn(xgbDf("probability"), lit(1))).drop("probability")

xgbDf.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("csv/XGBoostAudit")

pipelineModel.save("pipeline/XGBoostAudit")

//val pmmlBytes = new PMMLBuilder(df.schema, pipelineModel).buildByteArray()
//Files.write(Paths.get("pmml/XGBoostAudit.pmml"), pmmlBytes)
