import java.io.File

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMRegressor
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.jpmml.sparkml.{DatasetUtil, PipelineModelUtil, PMMLBuilder}
import org.jpmml.sparkml.feature.InvalidCategoryTransformer

var df = DatasetUtil.loadCsv(spark, new File("csv/AutoNA.csv"))

//DatasetUtil.storeSchema(df, new File("schema/AutoNA.json"))

val cat_cols = Array("cylinders", "model_year", "origin")
val cont_cols = Array("acceleration", "displacement", "horsepower", "weight")

val indexer = new StringIndexer().setInputCols(cat_cols).setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col)).setHandleInvalid("keep")
val indexTransformer = new InvalidCategoryTransformer().setInputCols(indexer.getOutputCols).setOutputCols(cat_cols.map(cat_col => "idxTransformed_" + cat_col))

val assembler = new VectorAssembler().setInputCols(indexTransformer.getOutputCols ++ cont_cols).setOutputCol("featureVector").setHandleInvalid("keep")

val regressor = new LightGBMRegressor().setNumIterations(101).setLabelCol("mpg").setFeaturesCol(assembler.getOutputCol)

val pipeline = new Pipeline().setStages(Array(indexer, indexTransformer, assembler, regressor))
val pipelineModel = pipeline.fit(df)

//PipelineModelUtil.storeZip(pipelineModel, new File("pipeline/LightGBMAutoNA.zip"))

new PMMLBuilder(df.schema, pipelineModel).buildFile(new File("pmml/LightGBMAutoNA.pmml"))

var lgbDf = pipelineModel.transform(df)
lgbDf = lgbDf.selectExpr("prediction as mpg")

DatasetUtil.storeCsv(lgbDf, new File("csv/LightGBMAutoNA.csv"))
