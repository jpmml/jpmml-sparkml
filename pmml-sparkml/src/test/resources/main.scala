import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.types.DataTypes
import org.jpmml.sparkml.DatasetUtil

class LibSVMTest extends SparkMLTest {

	override
	def load_iris(name: String): DataFrame = {
		var df = spark.read
			.format("libsvm")
			.load("libsvm/Iris.libsvm")

		df.withColumn("label", col("label").cast(DataTypes.IntegerType))
	}

	override
	def build_classification_pipeline(label_col: String, cat_cols: Array[String], cont_cols: Array[String], cat_encoding: CategoryEncoding): Pipeline = {
		val labelIndexer = new StringIndexer()
			.setInputCol(label_col)
			.setOutputCol("idx_" + label_col)

		val classifier = new LogisticRegression()
			.setLabelCol(labelIndexer.getOutputCol)
			.setFeaturesCol("features")

		new Pipeline()
			.setStages(Array(labelIndexer, classifier))
	}

	def run_iris(): Unit = {
		val label_col = "label"

		val df = load_iris("Iris")

		run_classification(df, label_col, null, null, null, "LogisticRegression", "IrisVec")
	}
}

val test = new LibSVMTest()
test.run_iris()