import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostRegressor}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DataType, FloatType, StructType}
import org.jpmml.sparkml.DatasetUtil
import org.jpmml.sparkml.feature.SparseToDenseTransformer

class XGBoostTest extends SparkMLTest {

	override
	def load_housing(name: String): DataFrame = {
		var df = super.load_iris(name)
		
		val schema = df.schema
		val floatSchema = DataType.fromJson(schema.json.replaceAll("integer", "float").replace("double", "float"))
		
		DatasetUtil.castColumns(df, floatSchema.asInstanceOf[StructType])
	}

	override
	def load_iris(name: String): DataFrame = {
		var df = super.load_iris(name)
		
		val schema = df.schema
		val floatSchema = DataType.fromJson(schema.json.replaceAll("double", "float"))
		
		DatasetUtil.castColumns(df, floatSchema.asInstanceOf[StructType])
	}

	override
	def build_features(cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding, withDomain: Boolean = true, maxCategories: Int = 100, dropLast: Boolean = false): Array[PipelineStage] = {
		val features = super.build_features(cat_cols, cont_cols, cat_encoding, withDomain, maxCategories, dropLast)

		cat_encoding match {
			case LEGACY_DIRECT_MIXED | LEGACY_OHE | MODERN_DIRECT => {
				// java.lang.IllegalArgumentException: We've detected sparse vectors in the dataset that need conversion to dense format. 
				val sparse2dense = new SparseToDenseTransformer()
					.setInputCol(features.last.asInstanceOf[HasOutputCol].getOutputCol)
					.setOutputCol("denseFeatureVec")

				features :+ sparse2dense
			}
			case _ => {
				features
			}
		}
	}

	override
	def build_classification_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		val labelIndexer = new StringIndexer()
			.setInputCol(label_col)
			.setOutputCol("idx_" + label_col)

		val features = build_features(cat_cols, cont_cols, cat_encoding, withDomain = false)

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

		cat_encoding match {
			case LEGACY_DIRECT_MIXED | MODERN_DIRECT => {
				classifier = classifier
					.setFeatureTypes(cat_cols.map(_ => "c") ++ cont_cols.map(_ => "q"))
			}
			case _ => ()
		}

		new Pipeline()
			.setStages(labelIndexer +: features :+ classifier)
	}

	override
	def build_regression_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		val features = build_features(cat_cols, cont_cols, cat_encoding, withDomain = false)

		val params = label_col match {
			case "docvis" =>
				Map("objective" -> "count:poisson", "num_round" -> 101)
			case _ =>
				Map("objective" -> "reg:squarederror", "num_round" -> 101)
		}

		var regressor = new XGBoostRegressor(params)
			.setLabelCol(label_col)
			.setFeaturesCol(features.last.asInstanceOf[HasOutputCol].getOutputCol)

		cat_encoding match {
			case LEGACY_DIRECT_MIXED | MODERN_DIRECT => {
				regressor = regressor
					.setFeatureTypes(cat_cols.map(_ => "c") ++ cont_cols.map(_ => "q"))
			}
			case _ => ()
		}

		new Pipeline()
			.setStages(features :+ regressor)
	}

	def run_audit(): Unit = {
		val label_col = "Adjusted"
		val cat_cols = Array("Education", "Employment", "Gender", "Marital", "Occupation")
		val cont_cols = Array("Age", "Hours", "Income")

		var df = load_audit("Audit")

		run_classification(df, label_col, cat_cols, cont_cols, CategoryEncoding.LEGACY_OHE, "XGBoost", "Audit", numericType = FloatType)

		df = load_audit("AuditNA")

		run_classification(df, label_col, cat_cols, cont_cols, CategoryEncoding.MODERN_DIRECT, "XGBoost", "AuditNA", numericType = FloatType)
	}

	def run_auto(): Unit = {
		val label_col = "mpg"
		val cat_cols = Array("cylinders", "model_year", "origin")
		val cont_cols = Array("acceleration", "displacement", "horsepower", "weight")

		var df = load_auto("Auto")

		run_regression(df, label_col, cat_cols, cont_cols, CategoryEncoding.LEGACY_OHE, "XGBoost", "Auto", numericType = FloatType)

		df = load_auto("AutoNA")

		run_regression(df, label_col, cat_cols, cont_cols, CategoryEncoding.MODERN_DIRECT, "XGBoost", "AutoNA", numericType = FloatType)
	}

	def run_housing(): Unit = {
		val label_col = "MEDV"
		val cat_cols = Array("CHAS", "RAD", "TAX")
		val cont_cols = Array("CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "PTRATIO", "B", "LSTAT")

		val df = load_housing("Housing")

		run_regression(df, label_col, cat_cols, cont_cols, CategoryEncoding.LEGACY_DIRECT_NUMERIC, "XGBoost", "Housing", numericType = FloatType)
	}

	def run_iris(): Unit = {
		val label_col = "Species"
		val cat_cols = Array[String]()
		val cont_cols = Array("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width")
		
		val df = load_iris("Iris")
		
		run_classification(df, label_col, cat_cols, cont_cols, null, "XGBoost", "Iris", numericType = FloatType)
	}

	def run_visit(): Unit = {
		val label_col = "docvis"
		val cat_cols = Array("edlevel", "female", "kids", "married", "outwork", "self")
		val cont_cols = Array("age", "educ", "hhninc")

		val df = load_visit("Visit")

		run_regression(df, label_col, cat_cols, cont_cols, CategoryEncoding.LEGACY_DIRECT_MIXED, "XGBoost", "Visit", numericType = FloatType)
	}
}

val test = new XGBoostTest()
test.run_audit()
test.run_auto()
test.run_housing()
test.run_iris()
test.run_visit()