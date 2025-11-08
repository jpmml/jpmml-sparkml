import java.io.File

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, StringIndexerModel, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DataType, DoubleType, FloatType, StringType}
import org.jpmml.sparkml.{DatasetUtil, PipelineModelUtil}
import org.jpmml.sparkml.feature.InvalidCategoryTransformer

object CategoryEncoding extends Enumeration {

	type CategoryEncoding = Value

	val LEGACY_DIRECT_NUMERIC, LEGACY_DIRECT_MIXED, LEGACY_OHE, MODERN_DIRECT = Value
}

import CategoryEncoding._

class SparkMLTest {

	def load_audit(name: String): DataFrame = {
		val df = DatasetUtil.loadCsv(spark, new File("csv/" + name + ".csv"))
		DatasetUtil.castColumn(df, "Adjusted", StringType)
	}

	def load_auto(name: String): DataFrame = {
		DatasetUtil.loadCsv(spark, new File("csv/" + name + ".csv"))
	}

	def load_housing(name: String): DataFrame = {
		DatasetUtil.loadCsv(spark, new File("csv/" + name + ".csv"))
	}

	def load_iris(name: String): DataFrame = {
		DatasetUtil.loadCsv(spark, new File("csv/" + name + ".csv"))
	}

	def build_features(cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Array[PipelineStage] = {
		cat_encoding match {
			case null => {
				val vecAssembler = new VectorAssembler()
					.setInputCols(cont_cols)
					.setOutputCol("featureVector")

				Array(vecAssembler)
			}

			case LEGACY_DIRECT_NUMERIC => {
				val vecAssembler = new VectorAssembler()
					.setInputCols(cat_cols ++ cont_cols)
					.setOutputCol("featureVec")

				val vecIndexer = new VectorIndexer()
					.setInputCol(vecAssembler.getOutputCol)
					.setOutputCol("indexedFeatureVec")

				Array(vecAssembler, vecIndexer)
			}

			case LEGACY_DIRECT_MIXED => {
				val stringIndexer = new StringIndexer()
					.setInputCols(cat_cols)
					.setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col))

				val vecAssembler = new VectorAssembler()
					.setInputCols(stringIndexer.getOutputCols ++ cont_cols)
					.setOutputCol("featureVec")

				Array(stringIndexer, vecAssembler)
			}

			case LEGACY_OHE => {
				var stringIndexer = new StringIndexer()
					.setInputCols(cat_cols)
					.setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col))

				val ohe = new OneHotEncoder()
					.setInputCols(stringIndexer.getOutputCols)
					.setOutputCols(cat_cols.map(cat_col => "ohe_" + cat_col))
					.setHandleInvalid("keep")
					.setDropLast(true)

				val vecAssembler = new VectorAssembler()
					.setInputCols(ohe.getOutputCols ++ cont_cols)
					.setOutputCol("featureVec")

				Array(stringIndexer, ohe, vecAssembler)
			}

			case MODERN_DIRECT => {
				var stringIndexer = new StringIndexer()
					.setInputCols(cat_cols)
					.setOutputCols(cat_cols.map(cat_col => "idx_" + cat_col))
					.setHandleInvalid("keep")

				val indexTransformer = new InvalidCategoryTransformer()
					.setInputCols(stringIndexer.getOutputCols)
					.setOutputCols(cat_cols.map(cat_col => "idxTransformed_" + cat_col))

				val vecAssembler = new VectorAssembler()
					.setInputCols(indexTransformer.getOutputCols ++ cont_cols)
					.setOutputCol("featureVector")
					.setHandleInvalid("keep")

				Array(stringIndexer, indexTransformer, vecAssembler)
			}
		}
	}

	def build_classification_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		???
	}

	def build_regression_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		???
	}

	def run_classification(df: DataFrame, label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding, algorithm: String, dataset: String, numericType: DataType = DoubleType): Unit = {
		val name = algorithm + dataset

		DatasetUtil.storeSchema(df, new File("schema/" + dataset + ".json"))

		val pipeline = build_classification_pipeline(label_col, cat_cols, cont_cols, cat_encoding)
		val pipelineModel = pipeline.fit(df)

		PipelineModelUtil.storeZip(pipelineModel, new File("pipeline/" + name + ".zip"))

		val indexerModel = pipelineModel.stages(0).asInstanceOf[StringIndexerModel]
		val labels = indexerModel.labelsArray.head

		val classLabel = udf {
			(value: Number) => labels(value.intValue())
		}

		val classProbability = numericType match {
			case FloatType => 
				udf { 
					(vec: Vector, index: Int) => vec(index).toFloat
				}
			case DoubleType => 
				udf {
					(vec: Vector, index: Int) => vec(index)
				}
			case _ =>
				throw new IllegalArgumentException()
		}

		var predDf = pipelineModel.transform(df)
			.select("prediction", "probability")

		predDf = predDf
			.withColumn(label_col, classLabel(col("prediction")))
			.drop("prediction")

		for(i <- labels.indices){
			val probabilityCol = "probability(" + labels(i) + ")"

			predDf = predDf.withColumn(probabilityCol, classProbability(col("probability"), lit(i)))
		}

		predDf = predDf
			.drop("probability")

		DatasetUtil.storeCsv(predDf, new File("csv/" + name + ".csv"))
	}

	def run_regression(df: DataFrame, label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding, algorithm: String, dataset: String, numericType: DataType = DoubleType): Unit = {
		val name = algorithm + dataset

		DatasetUtil.storeSchema(df, new File("schema/" + dataset + ".json"))

		val pipeline = build_regression_pipeline(label_col, cat_cols, cont_cols, cat_encoding)
		val pipelineModel = pipeline.fit(df)

		PipelineModelUtil.storeZip(pipelineModel, new File("pipeline/" + name + ".zip"))

		var predDf = pipelineModel.transform(df)
			.selectExpr("prediction as " + label_col)

		predDf = DatasetUtil.castColumn(predDf, label_col, numericType)

		DatasetUtil.storeCsv(predDf, new File("csv/" + name + ".csv"))
	}
}