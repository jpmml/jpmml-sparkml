Launch `spark-shell`:

```bash
export SPARK_HOME=/opt/spark-3.4.4-bin-hadoop3/
$SPARK_HOME/bin/spark-shell --jars ../../../../pmml-sparkml-example/target/pmml-sparkml-example-executable-3.0-SNAPSHOT.jar --packages ml.dmlc:xgboost4j-spark_2.12:3.1.0
```

Load the `main.scala` script:

```spark-shell
:load main.scala
```
