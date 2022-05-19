JPMML-SparkML [![Build Status](https://github.com/jpmml/jpmml-sparkml/workflows/maven/badge.svg)](https://github.com/jpmml/jpmml-sparkml/actions?query=workflow%3A%22maven%22)
=============

Java library and command-line application for converting Apache Spark ML pipelines to PMML.

# Table of Contents #

* [Features](#features)
  * [Overview](#overview)
  * [Supported libraries](#supported-libraries)
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

### Overview

* Functionality:
  * Thorough collection, analysis and encoding of feature information:
    * Names.
    * Data and operational types.
    * Valid, invalid and missing value spaces.
  * Pipeline extensions:
    * Pruning.
    * Model verification.
  * Conversion options.
* Extensibility:
  * Rich Java APIs for developing custom converters.
  * Automatic discovery and registration of custom converters based on `META-INF/sparkml2pmml.properties` resource files.
  * Direct interfacing with other JPMML conversion libraries such as [JPMML-LightGBM](https://github.com/jpmml/jpmml-lightgbm) and [JPMML-XGBoost](https://github.com/jpmml/jpmml-xgboost).
* Production quality:
  * Complete test coverage.
  * Fully compliant with the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library.

### Supported libraries

<details>
  <summary>Apache Spark ML</summary>

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
    * [`feature.OneHotEncoderModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/OneHotEncoderModel.html) (the result of fitting a `feature.OneHotEncoder`)
    * [`feature.PCAModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/PCAModel.html) (the result of fitting a `feature.PCA`)
    * [`feature.QuantileDiscretizer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/QuantileDiscretizer.html)
    * [`feature.RegexTokenizer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/RegexTokenizer.html)
    * [`feature.RFormulaModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/RFormulaModel.html) (the result of fitting a `feature.RFormula`)
    * [`feature.SQLTransformer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/SQLTransformer.html)
      * Subqueries.
      * Control flow expressions `case when` and `if`.
      * Arithmetic operators `+`, `-`, `*` and `/`.
      * Comparison operators `<`, `<=`, `==`, `>=` and `>`.
      * Logical operators `and`, `or` and `not`.
      * Math functions `abs`, `ceil`, `exp`, `expm1`, `floor`, `hypot`, `ln`, `log10`, `log1p`, `pow` and `rint`.
      * Trigonometric functions `sin`, `asin`, `sinh`, `cos`, `acos`, `cosh`, `tan`, `atan`, `tanh`.
      * Aggregation functions `greatest` and `least`.
      * RegExp functions `regexp_replace` and `rlike`.
      * String functions `char_length`, `character_length`, `concat`, `lcase`, `length`, `lower`, `substring`, `trim`, `ucase` and `upper`.
      * Type cast functions `boolean`, `cast`, `double`, `int` and `string`.
      * Value functions `in`, `isnan`, `isnull`, `isnotnull`, `negative` and `positive`.
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
    * [`fpm.FPGrowthModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/fpm/FPGrowthModel.html)
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
</details>

<details>
  <summary>LightGBM</summary>

  * Prediction models:
    * [`com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassificationModel`](https://mmlspark.blob.core.windows.net/docs/0.9.5/scala/com/microsoft/azure/synapse/ml/lightgbm/LightGBMClassificationModel.html)
    * [`com.microsoft.azure.synapse.ml.lightgbm.LightGBMRegressionModel`](https://mmlspark.blob.core.windows.net/docs/0.9.5/scala/com/microsoft/azure/synapse/ml/lightgbm/LightGBMRegressionModel.html)
</details>

<details>
  <summary>XGBoost</summary>

  * Feature transformers:
    * `org.jpmml.sparkml.xgboost.SparseToDenseTransformer`
  * Prediction models:
    * [`ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel`](https://xgboost.readthedocs.io/en/latest/jvm/scaladocs/xgboost4j-spark/ml/dmlc/xgboost4j/scala/spark/XGBoostClassificationModel.html)
    * [`ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel`](https://xgboost.readthedocs.io/en/latest/jvm/scaladocs/xgboost4j-spark/ml/dmlc/xgboost4j/scala/spark/XGBoostRegressionModel.html)
</details>

# Prerequisites #

* Apache Spark 1.5.X, 1.6.X, 2.0.X, 2.1.X, 2.2.X, 2.3.X, 2.4.X, 3.0.X, 3.1.X or 3.2.X.

# Installation #

### Library

JPMML-SparkML library JAR file (together with accompanying Java source and Javadocs JAR files) is released via [Maven Central Repository](https://repo1.maven.org/maven2/org/jpmml/).

The current version is **2.2.0** (19 May, 2022).

```xml
<dependency>
	<groupId>org.jpmml</groupId>
	<artifactId>pmml-sparkml</artifactId>
	<version>2.2.0</version>
</dependency>
```

### Compatibility matrix

Active development branches:

| Apache Spark version | JPMML-SparkML branch |
|----------------------|----------------------|
| 3.0.X | [`2.0.X`](https://github.com/jpmml/jpmml-sparkml/tree/2.0.X) |
| 3.1.X | [`2.1.X`](https://github.com/jpmml/jpmml-sparkml/tree/2.1.X) |
| 3.2.X | [`master`](https://github.com/jpmml/jpmml-sparkml/tree/master) |

Archived development branches:

| Apache Spark version | JPMML-SparkML branch |
|----------------------|----------------------|
| 1.5.X and 1.6.X | [`1.0.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.0.X) |
| 2.0.X | [`1.1.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.1.X) |
| 2.1.X | [`1.2.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.2.X) |
| 2.2.X | [`1.3.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.3.X) |
| 2.3.X | [`1.4.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.4.X) |
| 2.4.X | [`1.5.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.5.X) |
| ~~3.0.X~~ | ~~[`1.6.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.6.X)~~ |
| ~~3.1.X~~ | ~~[`1.7.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.7.X)~~ |
| ~~3.2.X~~ | ~~[`1.8.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.8.X)~~ |

### Example application

Enter the project root directory and build using [Apache Maven](https://maven.apache.org/):
```
mvn clean install
```

The build produces two JAR files:
* `pmml-sparkml/target/pmml-sparkml-2.2-SNAPSHOT.jar` - Library JAR file.
* `pmml-sparkml-exampletarget/pmml-sparkml-example-executable-2.2-SNAPSHOT.jar` - Example application JAR file.

# Usage #

### Library

Fitting a Spark ML pipeline:
```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.RFormula

val irisData = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("Iris.csv")
val irisSchema = irisData.schema

val rFormula = new RFormula().setFormula("Species ~ .")
val dtClassifier = new DecisionTreeClassifier().setLabelCol(rFormula.getLabelCol).setFeaturesCol(rFormula.getFeaturesCol)
val pipeline = new Pipeline().setStages(Array(rFormula, dtClassifier))

val pipelineModel = pipeline.fit(irisData)
```

Converting the fitted Spark ML pipeline to an in-memory PMML class model object:
```scala
import org.jpmml.sparkml.PMMLBuilder

val pmml = new PMMLBuilder(irisSchema, pipelineModel).build()
```

The representation of individual Spark ML pipeline stages can be customized via conversion options:
```scala
import org.jpmml.sparkml.PMMLBuilder
import org.jpmml.sparkml.model.HasTreeOptions

val dtClassifierModel = pipelineModel.stages(1)

val pmml = new PMMLBuilder(irisSchema, pipelineModel).putOption(dtClassifierModel, HasTreeOptions.OPTION_COMPACT, false).putOption(dtClassifierModel, HasTreeOptions.OPTION_ESTIMATE_FEATURE_IMPORTANCES, true).build()
```

Viewing the in-memory PMML class model object:
```scala
import javax.xml.transform.stream.StreamResult
import org.jpmml.model.JAXBUtil

JAXBUtil.marshalPMML(pmml, new StreamResult(System.out))
```

### Example application

The example application JAR file contains an executable class `org.jpmml.sparkml.example.Main`, which can be used to convert a pair of serialized `org.apache.spark.sql.types.StructType` and `org.apache.spark.ml.PipelineModel` objects to PMML.

The example application JAR file does not include Apache Spark runtime libraries. Therefore, this executable class must be executed using Apache Spark's `spark-submit` helper script.

For example, converting a pair of Spark ML schema and pipeline serialization files `pmml-sparkml/src/test/resources/schema/Iris.json` and `pmml-sparkml/src/test/resources/pipeline/DecisionTreeIris.zip`, respectively, to a PMML file `DecisionTreeIris.pmml`:
```
spark-submit --master local --class org.jpmml.sparkml.example.Main pmml-sparkml-example/target/pmml-sparkml-example-executable-2.2-SNAPSHOT.jar --schema-input pmml-sparkml/src/test/resources/schema/Iris.json --pipeline-input pmml-sparkml/src/test/resources/pipeline/DecisionTreeIris.zip --pmml-output DecisionTreeIris.pmml
```

Getting help:
```
spark-submit --master local --class org.jpmml.sparkml.example.Main pmml-sparkml-example/target/pmml-sparkml-example-executable-2.2-SNAPSHOT.jar --help
```

# Documentation #

* [Converting logistic regression models to PMML documents](https://openscoring.io/blog/2020/01/19/converting_logistic_regression_pmml/#apache-spark)
* [Deploying Apache Spark ML pipeline models on Openscoring REST web service](https://openscoring.io/blog/2020/02/16/deploying_sparkml_pipeline_openscoring_rest/)
* [Converting Apache Spark ML pipeline models to PMML documents](https://openscoring.io/blog/2018/07/09/converting_sparkml_pipeline_pmml/)

# License #

JPMML-SparkML is licensed under the terms and conditions of the [GNU Affero General Public License, Version 3.0](https://www.gnu.org/licenses/agpl-3.0.html).

If you would like to use JPMML-SparkML in a proprietary software project, then it is possible to enter into a licensing agreement which makes JPMML-SparkML available under the terms and conditions of the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause) instead.

# Additional information #

JPMML-SparkML is developed and maintained by Openscoring Ltd, Estonia.

Interested in using [Java PMML API](https://github.com/jpmml) software in your company? Please contact [info@openscoring.io](mailto:info@openscoring.io)