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

For a full list of supported transformer and estimator classes see the [`features.md`](features.md) file.

# Prerequisites #

* Apache Spark 3.0.X, 3.1.X, 3.2.X, 3.3.X, 3.4.X, 3.5.X or 4.0.X.

# Installation #

### Library

JPMML-SparkML library JAR file (together with accompanying Java source and Javadocs JAR files) is released via [Maven Central Repository](https://repo1.maven.org/maven2/org/jpmml/).

The current version is **3.2.3** (10 November, 2025).

```xml
<dependency>
	<groupId>org.jpmml</groupId>
	<artifactId>pmml-sparkml</artifactId>
	<version>3.2.3</version>
</dependency>
```

### Compatibility matrix

Active development branches:

| JPMML-SparkML branch | Apache Spark version |
|----------------------|----------------------|
| [`3.0.X`](https://github.com/jpmml/jpmml-sparkml/tree/3.0.X) | 3.4.X |
| [`3.1.X`](https://github.com/jpmml/jpmml-sparkml/tree/3.1.X) | 3.5.X |
| [`master`](https://github.com/jpmml/jpmml-sparkml/tree/master) | 4.0.X |

Stale development branches:

| JPMML-SparkML branch | Apache Spark version |
|----------------------|----------------------|
| [`2.0.X`](https://github.com/jpmml/jpmml-sparkml/tree/2.0.X) | 3.0.X |
| [`2.1.X`](https://github.com/jpmml/jpmml-sparkml/tree/2.1.X) | 3.1.X |
| [`2.2.X`](https://github.com/jpmml/jpmml-sparkml/tree/2.2.X) | 3.2.X |
| [`2.3.X`](https://github.com/jpmml/jpmml-sparkml/tree/2.3.X) | 3.3.X |
| [`2.4.X`](https://github.com/jpmml/jpmml-sparkml/tree/2.4.X) | 3.4.X |
| [`2.5.X`](https://github.com/jpmml/jpmml-sparkml/tree/2.5.X) | 3.5.X |

Archived development branches:

| JPMML-SparkML branch | Apache Spark version |
|----------------------|----------------------|
| [`1.0.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.0.X) | 1.5.X and 1.6.X |
| [`1.1.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.1.X) | 2.0.X |
| [`1.2.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.2.X) | 2.1.X |
| [`1.3.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.3.X) | 2.2.X |
| [`1.4.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.4.X) | 2.3.X |
| [`1.5.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.5.X) | 2.4.X |
| ~~[`1.6.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.6.X)~~ | ~~3.0.X~~ |
| ~~[`1.7.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.7.X)~~ | ~~3.1.X~~ |
| ~~[`1.8.X`](https://github.com/jpmml/jpmml-sparkml/tree/1.8.X)~~ | ~~3.2.X~~ |

### Example application

Enter the project root directory and build using [Apache Maven](https://maven.apache.org/):
```
mvn clean install
```

The build produces two JAR files:
* `pmml-sparkml/target/pmml-sparkml-3.2-SNAPSHOT.jar` - Library JAR file.
* `pmml-sparkml-exampletarget/pmml-sparkml-example-executable-3.2-SNAPSHOT.jar` - Example application JAR file.

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
spark-submit --master local --class org.jpmml.sparkml.example.Main pmml-sparkml-example/target/pmml-sparkml-example-executable-3.2-SNAPSHOT.jar --schema-input pmml-sparkml/src/test/resources/schema/Iris.json --pipeline-input pmml-sparkml/src/test/resources/pipeline/DecisionTreeIris.zip --pmml-output DecisionTreeIris.pmml
```

Getting help:
```
spark-submit --master local --class org.jpmml.sparkml.example.Main pmml-sparkml-example/target/pmml-sparkml-example-executable-3.2-SNAPSHOT.jar --help
```

# Documentation #

* [Training PySpark LightGBM pipelines](https://openscoring.io/blog/2023/05/26/pyspark_lightgbm_pipeline/)
* [Converting logistic regression models to PMML documents](https://openscoring.io/blog/2020/01/19/converting_logistic_regression_pmml/#apache-spark)
* [Deploying Apache Spark ML pipeline models on Openscoring REST web service](https://openscoring.io/blog/2020/02/16/deploying_sparkml_pipeline_openscoring_rest/)
* [Converting Apache Spark ML pipeline models to PMML documents](https://openscoring.io/blog/2018/07/09/converting_sparkml_pipeline_pmml/)

# License #

JPMML-SparkML is licensed under the terms and conditions of the [GNU Affero General Public License, Version 3.0](https://www.gnu.org/licenses/agpl-3.0.html).

If you would like to use JPMML-SparkML in a proprietary software project, then it is possible to enter into a licensing agreement which makes JPMML-SparkML available under the terms and conditions of the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause) instead.

# Additional information #

JPMML-SparkML is developed and maintained by Openscoring Ltd, Estonia.

Interested in using [Java PMML API](https://github.com/jpmml) software in your company? Please contact [info@openscoring.io](mailto:info@openscoring.io)