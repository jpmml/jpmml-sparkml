Launch `spark-shell`:

```bash
$SPARK_HOME/bin/spark-shell --jars ../../../../pmml-sparkml-example/target/pmml-sparkml-example-executable-3.1-SNAPSHOT.jar --packages ml.dmlc:xgboost4j-spark_2.12:${xgboost4j-spark.version}
```

Load scripts:

```spark-shell
:load ../../../../pmml-sparkml/src/test/resources/common.scala
:load main.scala
```
