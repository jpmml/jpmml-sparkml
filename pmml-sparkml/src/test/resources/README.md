Run `spark-submit`:

```bash
$SPARK_HOME/bin/spark-submit --jars ../../../../pmml-sparkml-example/target/pmml-sparkml-example-executable-3.0-SNAPSHOT.jar main.py
```

Launch `spark-shell`:

```bash
$SPARK_HOME/bin/spark-shell --jars ../../../../pmml-sparkml-example/target/pmml-sparkml-example-executable-3.0-SNAPSHOT.jar
```

Load scripts:

```spark-shell
:load common.scala
:load main.scala
```
