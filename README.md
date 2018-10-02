JPMML-SparkML
=============

Java library and command-line application for converting Apache Spark ML pipelines to PMML.

# Features #

* Supported Spark ML `PipelineStage` types:
  * Feature extractors, transformers and selectors:
    * [`feature.Binarizer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Binarizer.html)
    * [`feature.Bucketizer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Bucketizer.html)
    * [`feature.ChiSqSelectorModel`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/ChiSqSelectorModel.html) (the result of fitting a `feature.ChiSqSelector`)
    * [`feature.ColumnPruner`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/ColumnPruner.html)
    * [`feature.CountVectorizerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/CountVectorizerModel.html) (the result of fitting a `feature.CountVectorizer`)
    * [`feature.IDFModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/IDFModel.html) (the result of fitting a `feature.IDF`)
    * [`feature.IndexToString`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/IndexToString.html)
    * [`feature.Interaction`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Interaction.html)
    * [`feature.MaxAbsScalerModel`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/MaxAbsScalerModel.html) (the result of fitting a `feature.MaxAbsScaler`)
    * [`feature.MinMaxScalerModel`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/MinMaxScalerModel.html) (the result of fitting a `feature.MinMaxScaler`)
    * [`feature.NGram`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/NGram.html)
    * [`feature.OneHotEncoder`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/OneHotEncoder.html)
    * [`feature.PCAModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/PCAModel.html) (the result of fitting a `feature.PCA`)
    * [`feature.QuantileDiscretizer`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/QuantileDiscretizer.html)
    * [`feature.RegexTokenizer`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/RegexTokenizer.html)
    * [`feature.RFormulaModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/RFormulaModel.html) (the result of fitting a `feature.RFormula`)
    * [`feature.SQLTransformer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/SQLTransformer.html)
    * [`feature.StandardScalerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StandardScalerModel.html) (the result of fitting a `feature.StandardScaler`)
    * [`feature.StopWordsRemover`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StopWordsRemover.html)
    * [`feature.StringIndexerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StringIndexerModel.html) (the result of fitting a `feature.StringIndexer`)
    * [`feature.Tokenizer`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Tokenizer.html)
    * [`feature.VectorAssembler`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorAssembler.html)
    * [`feature.VectorAttributeRewriter`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorAttributeRewriter.html)
    * [`feature.VectorIndexerModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorIndexerModel.html) (the result of fitting a `feature.VectorIndexer`)
    * [`feature.VectorSlicer`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorSlicer.html)
  * Prediction models:
    * [`classification.DecisionTreeClassificationModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/DecisionTreeClassificationModel.html)
    * [`classification.GBTClassificationModel`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/GBTClassificationModel.html)
    * [`classification.LogisticRegressionModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/LogisticRegressionModel.html)
    * [`classification.MultilayerPerceptronClassificationModel`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/MultilayerPerceptronClassificationModel.html)
    * [`classification.NaiveBayesModel`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/NaiveBayesModel.html)
    * [`classification.RandomForestClassificationModel`](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/RandomForestClassificationModel.html)
    * [`clustering.KMeansModel`](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/clustering/KMeansModel.html)
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

* Apache Spark version 1.5.X, 1.6.X or 2.0.X.

# Installation #

## Library ##

JPMML-SparkML library JAR file (together with accompanying Java source and Javadocs JAR files) is released via [Maven Central Repository](http://repo1.maven.org/maven2/org/jpmml/).

The current version is **1.1.20** (26 June, 2018).

```xml
<dependency>
	<groupId>org.jpmml</groupId>
	<artifactId>jpmml-sparkml</artifactId>
	<version>1.1.20</version>
</dependency>
```

Compatibility matrix:

| JPMML-SparkML version | Apache Spark version | PMML version |
|-----------------------|----------------------|--------------|
| 1.0.0 through 1.0.9 | 1.5.X and 1.6.X | 4.2 |
| 1.1.0 | 2.0.X | 4.2 |
| 1.1.1 through 1.1.20 | 2.0.X | 4.3 |

JPMML-SparkML depends on the latest and greatest version of the [JPMML-Model](https://github.com/jpmml/jpmml-model) library, which is in conflict with the legacy version that is part of the Apache Spark distribution.

This conflict is documented in [SPARK-15526](https://issues.apache.org/jira/browse/SPARK-15526).

### Modifying Apache Spark installation ###

The embodiment of the legacy version of the JPMML-Model library:

* `$SPARK_HOME/jars/pmml-model-1.2.15.jar`
* `$SPARK_HOME/jars/pmml-schema-1.2.15.jar`

Removing these two JAR files will solve all conflicts for all applications forever.

### Compile-time conflict resolution ###

Excluding the legacy version of the JPMML-Model library:
```xml
<dependency>
	<groupId>org.apache.spark</groupId>
	<artifactId>spark-mllib_2.11</artifactId>
	<version>${spark.version}</version>
	<scope>provided</scope>
	<exclusions>
		<exclusion>
			<groupId>org.jpmml</groupId>
			<artifactId>pmml-model</artifactId>
		</exclusion>
	</exclusions>
</dependency>
```

### Run-time conflict resolution ###

Using the [Maven Shade Plugin](https://maven.apache.org/plugins/maven-shade-plugin/) to relocate all `org.dmg.pmml.*` and `org.jpmml.*` classes of the latest and greatest version of the JPMML-Model library to a different namespace (aka "shading"):
```xml
<plugin>
	<groupId>org.apache.maven.plugins</groupId>
	<artifactId>maven-shade-plugin</artifactId>
	<version>${maven.shade.version}</version>
	<executions>
		<execution>
			<phase>package</phase>
			<goals>
				<goal>shade</goal>
			</goals>
			<configuration>
				<relocations>
					<relocation>
						<pattern>org.dmg.pmml</pattern>
						<shadedPattern>org.shaded.dmg.pmml</shadedPattern>
					</relocation>
					<relocation>
						<pattern>org.jpmml</pattern>
						<shadedPattern>org.shaded.jpmml</shadedPattern>
					</relocation>
				</relocations>
			</configuration>
		</execution>
	</executions>
</plugin>
```

The downside of shading is that such relocated classes are incompatible with other JPMML APIs. For example, the `PMMLBuilder#build()` builder method would start returning `org.shaded.dmg.pmml.PMML` object instances, which are not valid substitutes for `org.dmg.pmml.PMML` object instances.

## Example application ##

Enter the project root directory and build using [Apache Maven](http://maven.apache.org/):
```
mvn clean install
```

The build produces two JAR files:
* `target/jpmml-sparkml-1.1-SNAPSHOT.jar` - Library JAR file.
* `target/jpmml-sparkml-executable-1.1-SNAPSHOT.jar` - Example application JAR file.

# Usage #

## Library ##

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

Please refer to the following resources for more ideas and code examples:

* [Converting Apache Spark ML pipeline models to PMML](http://openscoring.io/blog/2018/07/09/converting_sparkml_pipeline_pmml/)

## Example application ##

The example application JAR file contains an executable class `org.jpmml.sparkml.Main`, which can be used to convert a pair of serialized `org.apache.spark.sql.types.StructType` and `org.apache.spark.ml.PipelineModel` objects to PMML.

The example application JAR file does not include Apache Spark runtime libraries. Therefore, this executable class must be executed using Apache Spark's `spark-submit` helper script.

For example, converting a pair of Spark ML schema and pipeline serialization files `src/test/resources/schema/Iris.json` and `src/test/resources/pipeline/DecisionTreeIris.zip`, respectively, to a PMML file `DecisionTreeIris.pmml`:
```
spark-submit --master local --class org.jpmml.sparkml.Main target/jpmml-sparkml-executable-1.1-SNAPSHOT.jar --schema-input src/test/resources/schema/Iris.json --pipeline-input src/test/resources/pipeline/DecisionTreeIris.zip --pmml-output DecisionTreeIris.pmml
```

Getting help:
```
spark-submit --master local --class org.jpmml.sparkml.Main target/jpmml-sparkml-executable-1.1-SNAPSHOT.jar --help
```

# License #

JPMML-SparkML is dual-licensed under the [GNU Affero General Public License (AGPL) version 3.0](http://www.gnu.org/licenses/agpl-3.0.html), and a commercial license.

# Additional information #

JPMML-SparkML is developed and maintained by Openscoring Ltd, Estonia.

Interested in using JPMML software in your application? Please contact [info@openscoring.io](mailto:info@openscoring.io)
