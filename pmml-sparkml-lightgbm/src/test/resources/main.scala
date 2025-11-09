import com.microsoft.azure.synapse.ml.lightgbm.{LightGBMClassifier, LightGBMRegressor}
import org.apache.spark.ml.Pipeline
import org.jpmml.sparkml.DatasetUtil

class LightGBMTest extends SparkMLTest {

	override
	def build_classification_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		val labelIndexer = new StringIndexer()
			.setInputCol(label_col)
			.setOutputCol("idx_" + label_col)

		val features = build_features(cat_cols, cont_cols, cat_encoding)

		var classifier = new LightGBMClassifier()
			.setLabelCol(labelIndexer.getOutputCol)
			.setFeaturesCol(features.last.asInstanceOf[HasOutputCol].getOutputCol)

		label_col match {
			case "Adjusted" =>
				classifier = classifier
					.setObjective("binary")
					.setNumIterations(101)
			case "Species" =>
				classifier = classifier
					.setObjective("multiclass")
					.setNumIterations(17)
			case _ =>
				throw new IllegalArgumentException()
		}

		new Pipeline()
			.setStages(labelIndexer +: features :+ classifier)
	}

	override
	def build_regression_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		val features = build_features(cat_cols, cont_cols, cat_encoding)

		val regressor = new LightGBMRegressor()
			.setLabelCol(label_col)
			.setFeaturesCol(features.last.asInstanceOf[HasOutputCol].getOutputCol)
			.setNumIterations(101)

		new Pipeline()
			.setStages(features :+ regressor)
	}

	def run_audit(): Unit = {
		val label_col = "Adjusted"
		val cat_cols = Array("Education", "Employment", "Gender", "Marital", "Occupation")
		val cont_cols = Array("Age", "Hours", "Income")

		var df = load_audit("Audit")

		run_classification(df, label_col, cat_cols, cont_cols, CategoryEncoding.LEGACY_DIRECT_MIXED, "LightGBM", "Audit")

		df = load_audit("AuditNA")

		run_classification(df, label_col, cat_cols, cont_cols, CategoryEncoding.MODERN_DIRECT, "LightGBM", "AuditNA")
	}

	def run_auto(): Unit = {
		val label_col = "mpg"
		val cat_cols = Array("cylinders", "model_year", "origin")
		val cont_cols = Array("acceleration", "displacement", "horsepower", "weight")

		var df = load_auto("Auto")

		run_regression(df, label_col, cat_cols, cont_cols, CategoryEncoding.LEGACY_DIRECT_MIXED, "LightGBM", "Auto")

		df = load_auto("AutoNA")

		run_regression(df, label_col, cat_cols, cont_cols, CategoryEncoding.MODERN_DIRECT, "LightGBM", "AutoNA")
	}

	def run_iris(): Unit = {
		val label_col = "Species"
		val cat_cols = Array[String]()
		val cont_cols = Array("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width")
		
		val df = load_iris("Iris")
		
		run_classification(df, label_col, cat_cols, cont_cols, null, "LightGBM", "Iris")
	}
}

val test = new LightGBMTest()
test.run_audit()
test.run_auto()
test.run_iris()
