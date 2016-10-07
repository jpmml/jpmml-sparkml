JPMML-SparkML
=============

Java library and command-line application for converting Apache Spark ML pipelines to PMML.

# Features #

* Supported Spark ML `PipelineStage` types:
  * Feature extractors, transformers and selectors:
    * [`feature.Binarizer`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Binarizer.html)
    * [`feature.Bucketizer`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/Bucketizer.html)
    * [`feature.ChiSqSelectorModel`] (http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/ChiSqSelectorModel.html) (the result of fitting a `feature.ChiSqSelector`)
    * [`feature.ColumnPruner`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/ColumnPruner.html)
    * [`feature.IndexToString`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/IndexToString.html)
    * [`feature.MinMaxScalerModel`] (http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/MinMaxScalerModel.html) (the result of fitting a `feature.MinMaxScaler`)
    * [`feature.OneHotEncoder`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/OneHotEncoder.html)
    * [`feature.PCAModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/PCAModel.html) (the result of fitting a `feature.PCA`)
    * [`feature.QuantileDiscretizer`] (http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/QuantileDiscretizer.html)
    * [`feature.RFormulaModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/RFormulaModel.html) (the result of fitting a `feature.RFormula`)
    * [`feature.StandardScalerModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StandardScalerModel.html) (the result of fitting a `feature.StandardScaler`)
    * [`feature.StringIndexerModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/StringIndexerModel.html) (the result of fitting a `feature.StringIndexer`)
    * [`feature.VectorAssembler`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorAssembler.html)
    * [`feature.VectorAttributeRewriter`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorAttributeRewriter.html)
    * [`feature.VectorIndexerModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorIndexerModel.html) (the result of fitting a `feature.VectorIndexer`)
    * [`feature.VectorSlicer`] (http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/VectorSlicer.html)
  * Prediction models:
    * [`classification.DecisionTreeClassificationModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/DecisionTreeClassificationModel.html)
    * [`classification.GBTClassificationModel`] (http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/GBTClassificationModel.html)
    * [`classification.LogisticRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/LogisticRegressionModel.html)
    * [`classification.MultilayerPerceptronClassificationModel`] (http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/MultilayerPerceptronClassificationModel.html)
    * [`classification.RandomForestClassificationModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/classification/RandomForestClassificationModel.html)
    * [`clustering.KMeansModel`] (http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/clustering/KMeansModel.html)
    * [`regression.DecisionTreeRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/DecisionTreeRegressionModel.html)
    * [`regression.GBTRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/GBTRegressionModel.html)
    * [`regression.GeneralizedLinearRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/GeneralizedLinearRegressionModel.html)
    * [`regression.LinearRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/LinearRegressionModel.html)
    * [`regression.RandomForestRegressionModel`] (https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/regression/RandomForestRegressionModel.html)
  * Prediction model chains
* Production quality:
  * Complete test coverage.
  * Fully compliant with the [JPMML-Evaluator] (https://github.com/jpmml/jpmml-evaluator) library.

# Prerequisites #

* Apache Spark version 1.5.X, 1.6.X or 2.0.X.

# Installation #

## Library ##

JPMML-SparkML library JAR file (together with accompanying Java source and Javadocs JAR files) is released via [Maven Central Repository] (http://repo1.maven.org/maven2/org/jpmml/).

The current version is **1.1.3** (7 October, 2016).

```xml
<dependency>
	<groupId>org.jpmml</groupId>
	<artifactId>jpmml-sparkml</artifactId>
	<version>1.1.3</version>
</dependency>
```

Compatibility matrix:

| JPMML-SparkML version | Apache Spark version | PMML version |
|-----------------------|----------------------|--------------|
| 1.0.0 through 1.0.7 | 1.5.X and 1.6.X | 4.2 |
| 1.1.0 | 2.0.X | 4.2 |
| 1.1.1 through 1.1.3 | 2.0.X | 4.3 |

JPMML-SparkML depends on the latest and greatest version of the [JPMML-Model] (https://github.com/jpmml/jpmml-model) library, which is in conflict with the legacy version that is part of the Apache Spark distribution.

Excluding the legacy version of JPMML-Model library from the application classpath:
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

Using the [Maven Shade Plugin] (https://maven.apache.org/plugins/maven-shade-plugin/) for "shading" all the affected `org.dmg.pmml.*` and `org.jpmml.*` classes during the packaging of the application:
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

For a complete example, please see the [JPMML-SparkML-Bootstrap] (https://github.com/jpmml/jpmml-sparkml-bootstrap) project.

## Example application ##

Enter the project root directory and build using [Apache Maven] (http://maven.apache.org/):
```
mvn clean install
```

The build produces two JAR files:
* `target/jpmml-sparkml-1.1-SNAPSHOT.jar` - Library JAR file.
* `target/converter-executable-1.1-SNAPSHOT.jar` - Example application JAR file.

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

Converting the Spark ML pipeline to PMML using the `org.jpmml.sparkml.ConverterUtil#toPMML(StructType, PipelineModel)` utility method:
```java
PMML pmml = ConverterUtil.toPMML(schema, pipelineModel);

// Viewing the result
JAXBUtil.marshalPMML(pmml, new StreamResult(System.out));
```

## Example application ##

The example application JAR file contains an executable class `org.jpmml.sparkml.Main`, which can be used to convert a pair of serialized `org.apache.spark.sql.types.StructType` and `org.apache.spark.ml.PipelineModel` objects to PMML.

The example application JAR file does not include Apache Spark runtime libraries. Therefore, this executable class must be executed using Apache Spark's `spark-submit` helper script.

For example, converting a pair of Spark ML schema and pipeline serialization files `src/test/resources/ser/Iris.ser` and `src/test/resources/ser/DecisionTreeIris.ser`, respectively, to a PMML file `DecisionTreeIris.pmml`:
```
spark-submit --master local --class org.jpmml.sparkml.Main target/converter-executable-1.1-SNAPSHOT.jar --ser-schema-input src/test/resources/ser/Iris.ser --ser-pipeline-input src/test/resources/ser/DecisionTreeIris.ser --pmml-output DecisionTreeIris.pmml
```

Getting help:
```
spark-submit --master local --class org.jpmml.sparkml.Main target/converter-executable-1.1-SNAPSHOT.jar --help
```

# License #

JPMML-SparkML is licensed under the [GNU Affero General Public License (AGPL) version 3.0] (http://www.gnu.org/licenses/agpl-3.0.html). Other licenses are available on request.

# Additional information #

Please contact [info@openscoring.io] (mailto:info@openscoring.io)
