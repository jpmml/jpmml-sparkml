from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql.functions import udf, col, collect_set
from pyspark.sql.session import SparkSession
from pyspark.sql.types import DoubleType, StringType
from pyspark2pmml.ml.feature import CategoricalDomain, ContinuousDomain

import pandas
import shutil
import tempfile

def load_csv(name):
	return spark.read.csv("csv/" + name + ".csv", header = True, inferSchema = not name.endswith("NA"), nullValue = "N/A")

def store_schema(df, name):
	with open("schema/" + name + ".json", "w") as schema_file:
		schema_file.write(df.schema.json())

def store_csv(df, name):
	if not isinstance(df, pandas.DataFrame):
		df = df.toPandas()
	df.to_csv("csv/" + name + ".csv", index = False)

def store_zip(pipelineModel, name):
	with tempfile.TemporaryDirectory() as tmpDir:
		# See https://issues.apache.org/jira/browse/SPARK-28902
		pipelineModel._to_java().write().overwrite().save(tmpDir)
		shutil.make_archive("pipeline/" + name, format = "zip", root_dir = tmpDir)

def cast_col(df, name, type):
	df = df.withColumn("tmp_" + name, df[name].cast(type))
	df = df.drop(name)
	df = df.withColumnRenamed("tmp_" + name, name)

	return df

def _extract_probability(probabilities, index):
	if isinstance(probabilities, float):
		# No event
		if index == 0:
			return 1 - probabilities
		# Event
		elif index == 1:
			return probabilities
		else:
			raise ValueError()
	else:
		return float(probabilities[index])

def build_linearmodel_features(catCols, contCols, dropLast = False):
	catDomain = CategoricalDomain(missingValueTreatment = "asValue", missingValueReplacement = "(other)", inputCols = catCols, outputCols = ["domain" + catCol for catCol in catCols])
	contDomain = ContinuousDomain(missingValueTreatment = "asValue", missingValueReplacement = 0, inputCols = contCols, outputCols = ["domain" + contCol for contCol in contCols])

	stringIndexer = StringIndexer(inputCols = catDomain.getOutputCols(), outputCols = ["indexed" + catCol for catCol in catCols])
	ohe = OneHotEncoder(dropLast = dropLast, inputCols = stringIndexer.getOutputCols(), outputCols = ["ohe" + catCol for catCol in catCols])

	vecAssembler = VectorAssembler(inputCols = ohe.getOutputCols() + contDomain.getOutputCols(), outputCol = "featureVector")

	return [catDomain, contDomain, stringIndexer, ohe, vecAssembler]

def build_associationrules(df, pipeline, name):
	pipelineModel = pipeline.fit(df)
	store_zip(pipelineModel, name)

def build_clustering(df, pipeline, predictionCol, name):
	pipelineModel = pipeline.fit(df)
	store_zip(pipelineModel, name)

	dft = pipelineModel.transform(df)
	dft = dft[[predictionCol]]

	store_csv(dft, name)

def build_classification(df, pipeline, labelIndexerModel, predictionCol, probabilityCol, name):
	pipelineModel = pipeline.fit(df)
	store_zip(pipelineModel, name)

	dft = pipelineModel.transform(df)
	dft = dft[([predictionCol] if predictionCol else []) + ([probabilityCol] if probabilityCol else [])]

	labels = labelIndexerModel.labels

	if predictionCol:
		translate_label_udf = udf(lambda x: labels[int(x)], StringType())
		dft = dft.withColumn(labelIndexerModel.getInputCol(), translate_label_udf(col(predictionCol)))

		dft = dft.drop(predictionCol)
	elif probabilityCol:
		if len(labels) != 2:
			raise ValueError()
		extract_probability_udf = udf(lambda x: (labels[0] if _extract_probability(x, 1) < 0.5 else labels[1]), StringType())
		dft = dft.withColumn(labelIndexerModel.getInputCol(), extract_probability_udf(col(probabilityCol)))

	if probabilityCol:
		for index in range(0, len(labels)):
			extract_probability_udf = udf(lambda x: _extract_probability(x, index), DoubleType())
			dft = dft.withColumn("probability(" + labels[index] + ")", extract_probability_udf(col(probabilityCol)))

		dft = dft.drop(probabilityCol)

	store_csv(dft, name)

def build_regression(df, pipeline, label, predictionCol, name):
	pipelineModel = pipeline.fit(df)
	store_zip(pipelineModel, name)

	dft = pipelineModel.transform(df)
	dft = dft[[predictionCol]]

	dft = dft.withColumnRenamed(predictionCol, label)
	dft = dft.drop(predictionCol)

	store_csv(dft, name)

conf = SparkConf()
conf.setMaster("local").setAppName("main")
conf.set("spark.executor.allowSparkContext", "true")

sc = SparkContext(conf = conf)
sc.setLogLevel("WARN")

spark = SparkSession(sc)