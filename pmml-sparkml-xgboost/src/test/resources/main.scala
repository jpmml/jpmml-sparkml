import java.io.File

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostRegressor}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, StringIndexerModel, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DataType, FloatType, StringType, StructType}
import org.jpmml.sparkml.{DatasetUtil, PipelineModelUtil}
import org.jpmml.sparkml.feature.{InvalidCategoryTransformer, SparseToDenseTransformer}

object CategoryEncoding extends Enumeration {

	type CategoryEncoding = Value

	val LEGACY_DIRECT, LEGACY_OHE, MODERN_DIRECT = Value
}

import CategoryEncoding._

class XGBoostTest {

	def load_audit(name: String): DataFrame = {
		var df = DatasetUtil.loadCsv(spark, new File("csv/" + name + ".csv"))
		df = DatasetUtil.castColumn(df, "Adjusted", StringType)
		df
	}

	def load_auto(name: String): DataFrame = {
		var df = DatasetUtil.loadCsv(spark, new File("csv/" + name + ".csv"))
		df = DatasetUtil.castColumn(df, "origin", StringType)
		df
	}

	def load_housing(name: String): DataFrame = {
		val df = DatasetUtil.loadCsv(spark, new File("csv/" + name + ".csv"))
		df
	}

	def load_iris(name: String): DataFrame = {
		var df = DatasetUtil.loadCsv(spark, new File("csv/" + name + ".csv"))
		
		val schema = df.schema
		val floatSchema = DataType.fromJson(schema.json.replaceAll("double", "float"))
		
		df = DatasetUtil.castColumns(df, floatSchema.asInstanceOf[StructType])
		df
	}

	def build_features(cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Array[PipelineStage] = {
		cat_encoding match {
			case null => {
				val vecAssembler = new VectorAssembler()
					.setInputCols(cont_cols)
					.setOutputCol("featureVector")

				Array(vecAssembler)
			}

			case LEGACY_DIRECT => {
				val vecAssembler = new VectorAssembler()
					.setInputCols(cat_cols ++ cont_cols)
					.setOutputCol("featureVec")

				val vecIndexer = new VectorIndexer()
					.setInputCol(vecAssembler.getOutputCol)
					.setOutputCol("indexedFeatureVec")

				Array(vecAssembler, vecIndexer)
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

				val sparse2dense = new SparseToDenseTransformer()
					.setInputCol(vecAssembler.getOutputCol)
					.setOutputCol("denseFeatureVec")

				Array(stringIndexer, ohe, vecAssembler, sparse2dense)
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

				// java.lang.IllegalArgumentException: We've detected sparse vectors in the dataset that need conversion to dense format. 
				val sparse2dense = new SparseToDenseTransformer()
					.setInputCol(vecAssembler.getOutputCol)
					.setOutputCol("denseFeatureVec")

				Array(stringIndexer, indexTransformer, vecAssembler, sparse2dense)
			}
		}
	}

	def build_classification_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		val labelIndexer = new StringIndexer()
			.setInputCol(label_col)
			.setOutputCol("idx_" + label_col)

		val features = build_features(cat_cols, cont_cols, cat_encoding)

		val params = label_col match {
			case "Adjusted" => 
				Map("objective" -> "binary:logistic", "num_round" -> 101)
			case "Species" =>
				Map("objective" -> "multi:softprob", "num_class" -> 3, "num_round" -> 17)
			case _ =>
				throw new IllegalArgumentException()
		}

		var classifier = new XGBoostClassifier(params)
			.setLabelCol(labelIndexer.getOutputCol)
			.setFeaturesCol(features.last.asInstanceOf[HasOutputCol].getOutputCol)

		if(cat_encoding == CategoryEncoding.MODERN_DIRECT){
			classifier = classifier
				.setFeatureTypes(cat_cols.map(_ => "c") ++ cont_cols.map(_ => "q"))
		}

		new Pipeline()
			.setStages(labelIndexer +: features :+ classifier)
	}

	def build_regression_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		val features = build_features(cat_cols, cont_cols, cat_encoding)

		val params = Map("objective" -> "reg:squarederror", "num_round" -> 101)

		var regressor = new XGBoostRegressor(params)
			.setLabelCol(label_col)
			.setFeaturesCol(features.last.asInstanceOf[HasOutputCol].getOutputCol)

		if(cat_encoding == CategoryEncoding.MODERN_DIRECT){
			regressor = regressor
				.setFeatureTypes(cat_cols.map(_ => "c") ++ cont_cols.map(_ => "q"))
		}

		new Pipeline()
			.setStages(features :+ regressor)
	}

	def run_classification(df: DataFrame, label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding, name: String): Unit = {
		DatasetUtil.storeSchema(df, new File("schema/" + name + ".json"))

		val pipeline = build_classification_pipeline(label_col, cat_cols, cont_cols, cat_encoding)
		val pipelineModel = pipeline.fit(df)

		PipelineModelUtil.storeZip(pipelineModel, new File("pipeline/XGBoost" + name + ".zip"))

		val indexerModel = pipelineModel.stages(0).asInstanceOf[StringIndexerModel]
		val labels = indexerModel.labelsArray.head

		val classLabel = udf {
			(value: Float) => labels(value.toInt)
		}
		val classProbability = udf {
			(vec: Vector, index: Int) => vec(index).toFloat
		}

		var xgbDf = pipelineModel.transform(df)
			.select("prediction", "probability")

		xgbDf = xgbDf
			.withColumn(label_col, classLabel(col("prediction")))
			.drop("prediction")

		for(i <- labels.indices){
			val probabilityCol = "probability(" + labels(i) + ")"

			xgbDf = xgbDf.withColumn(probabilityCol, classProbability(col("probability"), lit(i)))
		}

		xgbDf = xgbDf
			.drop("probability")

		DatasetUtil.storeCsv(xgbDf, new File("csv/XGBoost" + name + ".csv"))
	}

	def run_regression(df: DataFrame, label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding, name: String): Unit = {
		DatasetUtil.storeSchema(df, new File("schema/" + name + ".json"))

		val pipeline = build_regression_pipeline(label_col, cat_cols, cont_cols, cat_encoding)
		val pipelineModel = pipeline.fit(df)

		PipelineModelUtil.storeZip(pipelineModel, new File("pipeline/XGBoost" + name + ".zip"))

		var xgbDf = pipelineModel.transform(df)
			.selectExpr("prediction as " + label_col)

		xgbDf = DatasetUtil.castColumn(xgbDf, label_col, FloatType)

		DatasetUtil.storeCsv(xgbDf, new File("csv/XGBoost" + name + ".csv"))
	}

	def run_audit(): Unit = {
		val label_col = "Adjusted"
		val cat_cols = Array("Education", "Employment", "Gender", "Marital", "Occupation")
		val cont_cols = Array("Age", "Hours", "Income")

		var df = load_audit("Audit")

		run_classification(df, label_col, cat_cols, cont_cols, CategoryEncoding.LEGACY_OHE, "Audit")

		df = load_audit("AuditNA")

		run_classification(df, label_col, cat_cols, cont_cols, CategoryEncoding.MODERN_DIRECT, "AuditNA")
	}

	def run_auto(): Unit = {
		val label_col = "mpg"
		val cat_cols = Array("cylinders", "model_year", "origin")
		val cont_cols = Array("acceleration", "displacement", "horsepower", "weight")

		var df = load_auto("Auto")

		run_regression(df, label_col, cat_cols, cont_cols, CategoryEncoding.LEGACY_OHE, "Auto")

		df = load_auto("AutoNA")

		run_regression(df, label_col, cat_cols, cont_cols, CategoryEncoding.MODERN_DIRECT, "AutoNA")
	}

	def run_housing(): Unit = {
		val label_col = "MEDV"
		val cat_cols = Array("CHAS", "RAD", "TAX")
		val cont_cols = Array("CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "PTRATIO", "B", "LSTAT")

		val df = load_housing("Housing")

		run_regression(df, label_col, cat_cols, cont_cols, CategoryEncoding.LEGACY_DIRECT, "Housing")
	}

	def run_iris(): Unit = {
		val label_col = "Species"
		val cat_cols = Array[String]()
		val cont_cols = Array("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width")
		
		val df = load_iris("Iris")
		
		run_classification(df, label_col, cat_cols, cont_cols, null, "Iris")
	}
}

val test = new XGBoostTest()

test.run_audit()
test.run_auto()
test.run_housing()
test.run_iris()