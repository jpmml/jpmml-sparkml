Launch `spark-shell`:

```bash
export SPARK_HOME=/opt/spark-3.4.4-bin-hadoop3/
$SPARK_HOME/bin/spark-shell --jars ../../../../pmml-sparkml-example/target/pmml-sparkml-example-executable-3.0-SNAPSHOT.jar --packages com.microsoft.azure:synapseml-lightgbm_2.12:1.0.15
```

Load scripts:

```spark-shell
:load ../../../../pmml-sparkml/src/test/resources/common.scala
:load main.scala
```
