import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types.DataTypes
import org.jpmml.sparkml.DatasetUtil

class LibSVMTest extends SparkMLTest {

	override
	def load_housing(name: String): DataFrame = {
		val df = spark.read
			.format("libsvm")
			.load("libsvm/Housing.libsvm")

		df
	}

	override
	def load_iris(name: String): DataFrame = {
		var df = spark.read
			.format("libsvm")
			.load("libsvm/Iris.libsvm")

		df.withColumn("label", col("label").cast(DataTypes.IntegerType))
	}

	override
	def build_regression_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		val stdScaler = new StandardScaler()
			.setInputCol("features")
			.setOutputCol("scaledFeatures")

		val regressor = new LinearRegression()
			.setLabelCol("label")
			.setFeaturesCol(stdScaler.getOutputCol)

		new Pipeline()
			.setStages(Array(stdScaler, regressor))
	}

	override
	def build_classification_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		val labelIndexer = new StringIndexer()
			.setInputCol(label_col)
			.setOutputCol("idx_" + label_col)

		val stdScaler = new StandardScaler()
			.setInputCol("features")
			.setOutputCol("scaledFeatures")

		val classifier = new LogisticRegression()
			.setLabelCol(labelIndexer.getOutputCol)
			.setFeaturesCol(stdScaler.getOutputCol)

		new Pipeline()
			.setStages(Array(labelIndexer, stdScaler, classifier))
	}

	def run_housing(): Unit = {
		val label_col = "label"

		val df = load_housing("Housing")

		run_regression(df, label_col, null, null, null, "LinearRegression", "HousingVec")
	}

	def run_iris(): Unit = {
		val label_col = "label"

		val df = load_iris("Iris")

		run_classification(df, label_col, null, null, null, "LogisticRegression", "IrisVec")
	}
}

val test = new LibSVMTest()
test.run_housing()
test.run_iris()