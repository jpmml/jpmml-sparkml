JPMML-SparkML
=============

Java library and command-line application for converting Apache Spark ML pipelines to PMML.

# Table of Contents #

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
  * [Library](#library)
  * [Example application](#example-application)
* [Usage](#usage)
  * [Library](#library-1)
  * [Example application](#example-application-1)
* [Documentation](#documentation)
* [License](#license)
* [Additional information](#additional-information)

# Features #

* Supported pipeline stage types:
  * Feature extractors, transformers and selectors:
    * [`feature.Binarizer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Binarizer.html)
    * [`feature.Bucketizer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Bucketizer.html)
    * [`feature.ChiSqSelectorModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/ChiSqSelectorModel.html) (the result of fitting a `feature.ChiSqSelector`)
    * [`feature.ColumnPruner`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/ColumnPruner.html)
    * [`feature.CountVectorizerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/CountVectorizerModel.html) (the result of fitting a `feature.CountVectorizer`)
    * [`feature.IDFModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/IDFModel.html) (the result of fitting a `feature.IDF`)
    * [`feature.ImputerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/ImputerModel.html) (the result of fitting a `feature.Imputer`)
    * [`feature.IndexToString`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/IndexToString.html)
    * [`feature.Interaction`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Interaction.html)
    * [`feature.MaxAbsScalerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/MaxAbsScalerModel.html) (the result of fitting a `feature.MaxAbsScaler`)
    * [`feature.MinMaxScalerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/MinMaxScalerModel.html) (the result of fitting a `feature.MinMaxScaler`)
    * [`feature.NGram`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/NGram.html)
    * [`feature.OneHotEncoder`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/OneHotEncoder.html)
    * [`feature.OneHotEncoderModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/OneHotEncoderModel.html) (the result of fitting a `feature.OneHotEncoderEstimator`)
    * [`feature.PCAModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/PCAModel.html) (the result of fitting a `feature.PCA`)
    * [`feature.QuantileDiscretizer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/QuantileDiscretizer.html)
    * [`feature.RegexTokenizer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/RegexTokenizer.html)
    * [`feature.RFormulaModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/RFormulaModel.html) (the result of fitting a `feature.RFormula`)
    * [`feature.SQLTransformer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/SQLTransformer.html)
      * Control flow expressions `case when` and `if`.
      * Arithmetic operators `+`, `-`, `*` and `/`.
      * Comparison operators `<`, `<=`, `==`, `>=` and `>`.
      * Logical operators `and`, `or` and `not`.
      * Math functions `abs`, `ceil`, `exp`, `floor`, `ln`, `log10`, `pow` and `rint`.
      * RegExp functions `regexp_replace` and `rlike`.
      * String functions `concat`, `lower`, `substring`, `trim` and `upper`.
      * Type cast functions `boolean`, `cast`, `double`, `int` and `string`.
      * Value functions `in`, `isnull`, `isnotnull`, `negative` and `positive`.
    * [`feature.StandardScalerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StandardScalerModel.html) (the result of fitting a `feature.StandardScaler`)
    * [`feature.StopWordsRemover`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StopWordsRemover.html)
    * [`feature.StringIndexerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StringIndexerModel.html) (the result of fitting a `feature.StringIndexer`)
    * [`feature.Tokenizer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Tokenizer.html)
    * [`feature.VectorAssembler`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorAssembler.html)
    * [`feature.VectorAttributeRewriter`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorAttributeRewriter.html)
    * [`feature.VectorIndexerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorIndexerModel.html) (the result of fitting a `feature.VectorIndexer`)
    * [`feature.VectorSizeHint`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorSizeHint.html)
    * [`feature.VectorSlicer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorSlicer.html)
  * Prediction models:
    * [`classification.DecisionTreeClassificationModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/DecisionTreeClassificationModel.html)
    * [`classification.GBTClassificationModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/GBTClassificationModel.html)
    * [`classification.LinearSVCModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/LinearSVCModel.html)
    * [`classification.LogisticRegressionModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/LogisticRegressionModel.html)
    * [`classification.MultilayerPerceptronClassificationModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/MultilayerPerceptronClassificationModel.html)
    * [`classification.NaiveBayesModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/NaiveBayesModel.html)
    * [`classification.RandomForestClassificationModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/RandomForestClassificationModel.html)
    * [`clustering.KMeansModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/clustering/KMeansModel.html)
    * [`regression.DecisionTreeRegressionModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/DecisionTreeRegressionModel.html)
    * [`regression.GBTRegressionModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/GBTRegressionModel.html)
    * [`regression.GeneralizedLinearRegressionModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/GeneralizedLinearRegressionModel.html)
    * [`regression.LinearRegressionModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/LinearRegressionModel.html)
    * [`regression.RandomForestRegressionModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/RandomForestRegressionModel.html)
  * Prediction model chains:
    * [`PipelineModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/PipelineModel.html)
    * Referencing the prediction column (`HasPredictionCol#getPredictionCol()`) of earlier clustering, classification and regression models.
    * Referencing the predicted probabilities column (`HasProbabilityCol#getProbabilityCol()`) of earlier classification models.
  * Hyperparameter selectors and tuners:
    * [`tuning.CrossValidatorModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/tuning/CrossValidatorModel.html)
    * [`tuning.TrainValidationSplitModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/tuning/TrainValidationSplitModel.html)
* Production quality:
  * Complete test coverage.
  * Fully compliant with the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library.

# Prerequisites #

* Apache Spark version 1.5.X, 1.6.X, 2.0.X, 2.1.X, 2.2.X, 2.3.X or 2.4.X.

# Installation #

### Library

JPMML-SparkML library JAR file (together with accompanying Java source and Javadocs JAR files) is released via [Maven Central Repository](https://repo1.maven.org/maven2/org/jpmml/).

The current version is **1.5.7** (11 January, 2020).

```xml
<dependency>
	<groupId>org.jpmml</groupId>
	<artifactId>jpmml-sparkml</artifactId>
	<version>1.5.7</version>
</dependency>
```

Compatibility matrix:

| Apache Spark version | JPMML-SparkML branch | Status |
|----------------------|----------------------|--------|
| 1.5.X and 1.6.X | [`1.0.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.0.X) | Archived |
| 2.0.X | [`1.1.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.1.X) | Archived |
| 2.1.X | [`1.2.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.2.X) | Archived |
| 2.2.X | [`1.3.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.3.X) | Archived |
| 2.3.X | [`1.4.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.4.X) | Active |
| 2.4.X | [`master`](https://github.com/jpmml/jpmml-sparkml/tree/master) | Active |

JPMML-SparkML depends on the latest and greatest version of the [JPMML-Model](https://github.com/jpmml/jpmml-model) library, which is in conflict with the legacy version that is part of Apache Spark version 2.0.X, 2.1.X and 2.2.X distributions.

This conflict is documented in [SPARK-15526](https://issues.apache.org/jira/browse/SPARK-15526). For possible resolutions, please switch from this README.md file to the README.md file of some earlier JPMML-SparkML development branch.

### Example application

Enter the project root directory and build using [Apache Maven](https://maven.apache.org/):
```
mvn clean install
```

The build produces two JAR files:
* `target/jpmml-sparkml-1.5-SNAPSHOT.jar` - Library JAR file.
* `target/jpmml-sparkml-executable-1.5-SNAPSHOT.jar` - Example application JAR file.

# Usage #

### Library

Fitting a Spark ML pipeline that only makes use of supported Transformer types:
```java
DataFrame irisData = ...;

StructType schema = irisData.schema();

RFormula formula = new RFormula()
	.setFormula("Species ~ .");

DecisionTreeClassifier classifier = new DecisionTreeClassifier()
	.setLabelCol(formula.getLabelCol())
	.setFeaturesCol(formula.getFeaturesCol());

Pipeline pipeline = new Pipeline()
	.setStages(new PipelineStage[]{formula, classifier});

PipelineModel pipelineModel = pipeline.fit(irisData);
```

Converting the Spark ML pipeline to PMML using the `org.jpmml.sparkml.PMMLBuilder` builder class:
```java
PMML pmml = new PMMLBuilder(schema, pipelineModel)
	.build();

// Viewing the result
JAXBUtil.marshalPMML(pmml, new StreamResult(System.out));
```

### Example application

The example application JAR file contains an executable class `org.jpmml.sparkml.Main`, which can be used to convert a pair of serialized `org.apache.spark.sql.types.StructType` and `org.apache.spark.ml.PipelineModel` objects to PMML.

The example application JAR file does not include Apache Spark runtime libraries. Therefore, this executable class must be executed using Apache Spark's `spark-submit` helper script.

For example, converting a pair of Spark ML schema and pipeline serialization files `src/test/resources/schema/Iris.json` and `src/test/resources/pipeline/DecisionTreeIris.zip`, respectively, to a PMML file `DecisionTreeIris.pmml`:
```
spark-submit --master local --class org.jpmml.sparkml.Main target/jpmml-sparkml-executable-1.5-SNAPSHOT.jar --schema-input src/test/resources/schema/Iris.json --pipeline-input src/test/resources/pipeline/DecisionTreeIris.zip --pmml-output DecisionTreeIris.pmml
```

Getting help:
```
spark-submit --master local --class org.jpmml.sparkml.Main target/jpmml-sparkml-executable-1.5-SNAPSHOT.jar --help
```

# Documentation #

* [Converting Apache Spark ML pipeline models to PMML documents](https://openscoring.io/blog/2018/07/09/converting_sparkml_pipeline_pmml/)

# License #

JPMML-SparkML is dual-licensed under the [GNU Affero General Public License (AGPL) version 3.0](https://www.gnu.org/licenses/agpl-3.0.html), and a commercial license.

# Additional information #

JPMML-SparkML is developed and maintained by Openscoring Ltd, Estonia.

Interested in using JPMML software in your application? Please contact [info@openscoring.io](mailto:info@openscoring.io)
