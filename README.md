JPMML-SparkML
=============

Java library and command-line application for converting Spark ML pipelines to PMML.

# Features #

* Supported Transformer types:
  * Feature transformers:
    * [`feature.OneHotEncoder`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/OneHotEncoder.html)
    * [`feature.VectorAssembler`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorAssembler.html)
  * Fitted feature transformers:
    * [`feature.StandardScalerModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StandardScalerModel.html)
    * [`feature.StringIndexerModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StringIndexerModel.html)
  * Prediction models:
    * [`classification.DecisionTreeClassificationModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/DecisionTreeClassificationModel.html)
    * [`classification.LogisticRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/LogisticRegressionModel.html)
    * [`classification.RandomForestClassificationModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/RandomForestClassificationModel.html)
    * [`regression.DecisionTreeRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/DecisionTreeRegressionModel.html)
    * [`regression.GBTRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/GBTRegressionModel.html)
    * [`regression.LinearRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/LinearRegressionModel.html)
    * [`regression.RandomForestRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/RandomForestRegressionModel.html)
* Production quality:
  * Complete test coverage.
  * Fully compliant with the [JPMML-Evaluator] (https://github.com/jpmml/jpmml-evaluator) library.

# Prerequisites #

* Apache Spark version 1.6.0 or newer.

# Installation #

Enter the project root directory and build using [Apache Maven] (http://maven.apache.org/):
```
mvn clean install
```

The build produces two JAR files:
* `target/jpmml-sparkml-1.0-SNAPSHOT.jar` - Library JAR file.
* `target/converter-executable-1.0-SNAPSHOT.jar` - Example application JAR file.

# Usage #

## Library ##

Adding the JPMML-SparkML dependency to the project:
```xml
<dependency>
	<groupId>org.jpmml</groupId>
	<artifactId>jpmml-sparkml</artifactId>
	<version>1.0-SNAPSHOT</version>
</dependency>
```

Fitting a Spark ML pipeline that only makes use of supported Transformer types:
```java
DataFrame irisData = ...;

StringIndexerModel speciesIndexer = new StringIndexer()
	.setInputCol("Species")
	.setOutputCol("speciesIndex")
	.fit(irisData);

VectorAssembler vectorAssembler = new VectorAssembler()
	.setInputCols(new String[]{"Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"})
	.setOutputCol("featureVector");

DecisionTreeClassifier classifier = new DecisionTreeClassifier()
	.setLabelCol(speciesIndexer.getOutputCol())
	.setFeaturesCol(vectorAssembler.getOutputCol());

IndexToString labelConverter = new IndexToString()
	.setInputCol(classifier.getPredictionCol())
	.setOutputCol("predictedSpecies")
	.setLabels(speciesIndexer.labels());

Pipeline pipeline = new Pipeline()
	.setStages(new PipelineStage[]{speciesIndexer, vectorAssembler, classifier, labelConverter});

PipelineModel pipelineModel = pipeline.fit(irisData);
```

Converting the Spark ML pipeline to PMML using the `org.jpmml.sparkml.PipelineModelUtil#toPMML(StructType, PipelineModel)` utility method:
```java
PMML pmml = PipelineModelUtil.toPMML(irisData.schema(), pipelineModel);

// Viewing the result
JAXBUtil.marshalPMML(pmml, new StreamResult(System.out));
```

Saving the Spark ML pipeline in Java serialization data format to a file `pipeline.ser` for conversion with the example application:
```java
try(OutputStream os = new FileOutputStream("pipeline.ser")){

	try(ObjectOutputStream oos = new ObjectOutputStream(os)){
		oos.writeObject(pipelineModel);
	}
}
```

## Example application ##

The example application JAR file contains an executable class `org.jpmml.sparkml.Main`, which can be used to convert serialized `org.apache.spark.ml.PipelineModel` objects to PMML.

The example application JAR file does not include Apache Spark runtime libraries. Therefore, this executable class must be executed using Apache Spark's `spark-submit` helper script.

Converting Spark ML schema and pipeline serialization files `schema.ser` and `pipeline.ser`, respectively, to a PMML file `pipeline.pmml`:
```
spark-submit --master local[1] --class org.jpmml.sparkml.Main target/converter-executable-1.0-SNAPSHOT.jar --ser-schema-input schema.ser --ser-pipeline-input pipeline.ser --pmml-output pipeline.pmml
```

Getting help:
```
spark-submit --master local[1] --class org.jpmml.sparkml.Main target/converter-executable-1.0-SNAPSHOT.jar --help
```

# License #

JPMML-SparkML is licensed under the [GNU Affero General Public License (AGPL) version 3.0] (http://www.gnu.org/licenses/agpl-3.0.html). Other licenses are available on request.

# Additional information #

Please contact [info@openscoring.io] (mailto:info@openscoring.io)
