import java.io.File

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.StringType
import org.jpmml.sparkml.{DatasetUtil, PipelineModelUtil, PMMLBuilder}

var df = DatasetUtil.loadCsv(spark, new File("csv/Iris.csv"))

//DatasetUtil.storeSchema(df, new File("schema/Iris.json"))

val labelIndexer = new StringIndexer().setInputCol("Species").setOutputCol("idx_Species")
val labelIndexerModel = labelIndexer.fit(df)

val assembler = new VectorAssembler().setInputCols(Array("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width")).setOutputCol("featureVector")

val classifier = new LightGBMClassifier().setObjective("multiclass").setNumIterations(17).setLabelCol(labelIndexer.getOutputCol).setFeaturesCol(assembler.getOutputCol)

val pipeline = new Pipeline().setStages(Array(labelIndexer, assembler, classifier))
val pipelineModel = pipeline.fit(df)

//PipelineModelUtil.storeZip(pipelineModel, new File("pipeline/LightGBMIris.zip"))

new PMMLBuilder(df.schema, pipelineModel).buildFile(new File("pmml/LightGBMIris.pmml"))

val predLabel = udf{ (value: Double) => labelIndexerModel.labels(value.toInt) }
val vectorToColumn = udf{ (vec: Vector, index: Int) => vec(index) }

var lgbDf = pipelineModel.transform(df)
lgbDf = lgbDf.selectExpr("prediction", "probability")
lgbDf = lgbDf.withColumn("Species", predLabel(lgbDf("prediction"))).drop("prediction")
lgbDf = lgbDf.withColumn("probability(setosa)", vectorToColumn(lgbDf("probability"), lit(0))).withColumn("probability(versicolor)", vectorToColumn(lgbDf("probability"), lit(1))).withColumn("probability(virginica)", vectorToColumn(lgbDf("probability"), lit(2))).drop("probability").drop("probability")

DatasetUtil.storeCsv(lgbDf, new File("csv/LightGBMIris.csv"))
