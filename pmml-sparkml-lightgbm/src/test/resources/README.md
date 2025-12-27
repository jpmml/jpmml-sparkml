Launch `spark-shell`:

```bash
$SPARK_HOME/bin/spark-shell --jars "../../../../pmml-sparkml-example/target/pmml-sparkml-example-executable-3.3-SNAPSHOT.jar,scala-library-2.12.20.jar" --packages com.microsoft.azure:synapseml-lightgbm_2.12:${synapseml-lightgbm.version}
```

Load scripts:

```spark-shell
:load ../../../../pmml-sparkml/src/test/resources/common.scala
:load main.scala
```
