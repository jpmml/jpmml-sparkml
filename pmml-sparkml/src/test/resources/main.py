from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, LinearSVC, LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import Binarizer, Bucketizer, ChiSqSelector, CountVectorizer, IDF, Imputer, Interaction, MaxAbsScaler, MinMaxScaler, NGram, OneHotEncoder, PCA, QuantileDiscretizer, RegexTokenizer, RFormula, SQLTransformer, StandardScaler, StopWordsRemover, StringIndexer, VectorAssembler, VectorIndexer, VectorSizeHint, VectorSlicer
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor, GeneralizedLinearRegression, IsotonicRegression, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.sql.types import BooleanType, DoubleType, IntegerType, StringType

import sys

from common import *

datasets = []

if __name__ == "__main__":
	if len(sys.argv) > 1:
		datasets = (sys.argv[1]).split(",")
	else:
		datasets = ["Audit", "Auto", "Housing", "Iris", "Sentiment", "Shopping", "Visit"]

def build_formula_audit(audit_df, name):
	adjustedIndexer = StringIndexer(inputCol = "Adjusted", outputCol = "adjustedIndex")
	adjustedIndexerModel = adjustedIndexer.fit(audit_df)

	sqlTransformer = SQLTransformer(statement = "SELECT *, (Income / (Hours * 52)) AS Hourly_Income FROM __THIS__")

	rFormula = RFormula(formula = "Adjusted ~ . + Education:Occupation + Hourly_Income:Occupation + Gender:Marital", labelCol = "Adjusted", featuresCol = "featureVector")

	lr = LogisticRegression(labelCol = rFormula.getLabelCol(), featuresCol = rFormula.getFeaturesCol())

	build_classification(audit_df, Pipeline(stages = [sqlTransformer, rFormula, lr]), adjustedIndexerModel, lr.getPredictionCol(), lr.getProbabilityCol(), name)

def build_modelchain_audit(audit_df, name):
	adjustedIndexer = StringIndexer(inputCol = "Adjusted", outputCol = "adjustedIndex")
	adjustedIndexerModel = adjustedIndexer.fit(audit_df)

	genderIndexer = StringIndexer(inputCol = "Gender", outputCol = "genderIndex")
	genderIndexerModel = genderIndexer.fit(audit_df)

	occupationIndexer = StringIndexer(inputCol = "Occupation", outputCol = "occupationIndex")
	occupationIndexerModel = occupationIndexer.fit(audit_df)

	genderVectorAssembler = VectorAssembler(inputCols = ["Income", "Hours"] + [occupationIndexer.getOutputCol()])

	genderDt = DecisionTreeClassifier(maxDepth = 3, labelCol = genderIndexer.getOutputCol(), featuresCol = genderVectorAssembler.getOutputCol(), predictionCol = "genderPrediction", probabilityCol = "genderProbability")

	genderSqlTransformer = SQLTransformer(statement = "SELECT Adjusted, genderPrediction, genderProbability, Age, Employment, Education, Marital, Deductions FROM __THIS__")

	genderVectorSlicer = VectorSlicer(indices = [1], inputCol = "genderProbability", outputCol = "slicedGenderProbability")

	adjustedFormula = RFormula(formula = "Adjusted ~ genderPrediction + slicedGenderProbability + Age + Employment + Education + Marital + Deductions", labelCol = "Adjusted", featuresCol = "adjustedFeatureVector")

	adjustedLr = LogisticRegression(labelCol = adjustedFormula.getLabelCol(), featuresCol = adjustedFormula.getFeaturesCol())

	build_classification(audit_df, Pipeline(stages = [genderIndexer, occupationIndexer, genderVectorAssembler, genderDt, genderSqlTransformer, genderVectorSlicer, adjustedFormula, adjustedLr]), adjustedIndexerModel, adjustedLr.getPredictionCol(), adjustedLr.getProbabilityCol(), name)

def build_classification_audit(audit_df, classifier, name):
	stages = []
	featureCols = []

	adjustedIndexer = StringIndexer(inputCol = "Adjusted", outputCol = "adjustedIndex")
	adjustedIndexerModel = adjustedIndexer.fit(audit_df)

	stages.append(adjustedIndexer)

	if isinstance(classifier, MultilayerPerceptronClassifier):
		ageIncomeVectorAssembler = VectorAssembler(inputCols = ["Age", "Income"], outputCol = "ageIncomeVector")
		ageIncomeScaler = StandardScaler(inputCol = ageIncomeVectorAssembler.getOutputCol(), outputCol = "scaledAgeIncome")

		stages += [ageIncomeVectorAssembler, ageIncomeScaler]
		featureCols.append(ageIncomeScaler.getOutputCol())
	else:
		ageIncomeInteraction = Interaction(inputCols = ["Age", "Income"], outputCol = "Age:Income")

		stages.append(ageIncomeInteraction)
		featureCols.append(ageIncomeInteraction.getOutputCol())

	hoursBinarizer = Binarizer(threshold = 35, inputCol = "Hours", outputCol = "binarizedHours")

	stages.append(hoursBinarizer)
	featureCols.append(hoursBinarizer.getOutputCol())

	stringCols = ["Employment", "Education", "Marital", "Occupation", "Gender"]
	for stringCol in stringCols:
		stringIndexer = StringIndexer(inputCol = stringCol, outputCol = stringCol.lower() + "Index")
		stringIndexerModel = stringIndexer.fit(audit_df)

		ohe = OneHotEncoder(dropLast = False, inputCol = stringIndexerModel.getOutputCol(), outputCol = stringCol.lower() + "ClassVector")

		if len(stringIndexerModel.labels) > 3:
			chiSqSelector = ChiSqSelector(numTopFeatures = 3, labelCol = adjustedIndexer.getOutputCol(), featuresCol = ohe.getOutputCol(), outputCol = stringCol.lower() + "SelectedClassVector")

			stages.append(Pipeline(stages = [stringIndexerModel, ohe, chiSqSelector]))
			featureCols.append(chiSqSelector.getOutputCol())
		else:
			stages.append(Pipeline(stages = [stringIndexerModel, ohe]))
			featureCols.append(ohe.getOutputCol())

	if isinstance(classifier, MultilayerPerceptronClassifier):
		pass
	else:
		employmentOccupationInteraction = Interaction(inputCols = ["employmentSelectedClassVector", "occupationSelectedClassVector"], outputCol = "Employment:Occupation")

		stages.append(employmentOccupationInteraction)
		featureCols.append(employmentOccupationInteraction.getOutputCol())

	vectorAssembler = VectorAssembler(inputCols = featureCols, outputCol = "featureVector")

	classifier = classifier.setLabelCol(adjustedIndexer.getOutputCol()).setFeaturesCol(vectorAssembler.getOutputCol())

	stages += [vectorAssembler, classifier]

	pipeline = Pipeline(stages = stages)

	if isinstance(classifier, DecisionTreeClassifier):
		paramGrid = ParamGridBuilder() \
			.addGrid(classifier.maxDepth, [5, 6, 7]) \
			.addGrid(classifier.minInstancesPerNode, [5, 10]) \
			.build()

		classificationEvaluator = BinaryClassificationEvaluator(labelCol = classifier.getLabelCol(), rawPredictionCol = classifier.getRawPredictionCol())

		trainValidationSplit = TrainValidationSplit(estimator = pipeline, estimatorParamMaps = paramGrid, evaluator = classificationEvaluator, trainRatio = 0.75, seed = 63317)

		pipeline = Pipeline(stages = [trainValidationSplit])

	if isinstance(classifier, GeneralizedLinearRegression):
		predictionCol = None
		probabilityCol = classifier.getPredictionCol()
	else:
		predictionCol = classifier.getPredictionCol()
		probabilityCol = classifier.getProbabilityCol()

	build_classification(audit_df, pipeline, adjustedIndexerModel, predictionCol, probabilityCol, name)

if "Audit" in datasets:
	audit_df = load_csv("Audit")
	print(audit_df.dtypes)

	audit_df = cast_col(audit_df, "Deductions", BooleanType())
	audit_df = cast_col(audit_df, "Hours", DoubleType())
	print(audit_df.dtypes)

	store_schema(audit_df, "Audit")

	build_formula_audit(audit_df, "LogisticRegressionAudit")
	build_modelchain_audit(audit_df, "ModelChainAudit")

	build_classification_audit(audit_df, DecisionTreeClassifier(), "DecisionTreeAudit")
	build_classification_audit(audit_df, GBTClassifier(), "GBTAudit")
	build_classification_audit(audit_df, GeneralizedLinearRegression(family = "binomial", link = "logit"), "GLMAudit")
	build_classification_audit(audit_df, MultilayerPerceptronClassifier(layers = [17, 2 * 17, 2], maxIter = 100, seed = 13), "NeuralNetworkAudit")
	build_classification_audit(audit_df, NaiveBayes(), "NaiveBayesAudit")
	build_classification_audit(audit_df, RandomForestClassifier(numTrees = 13, minInstancesPerNode = 20), "RandomForestAudit")

def build_classification_auditna(audit_df, name):
	adjustedIndexer = StringIndexer(inputCol = "Adjusted", outputCol = "adjustedIndex")
	adjustedIndexerModel = adjustedIndexer.fit(audit_df)

	catCols = ["Deductions", "Education", "Employment", "Gender", "Marital", "Occupation"]
	contCols = ["Age", "Hours", "Income"]

	features = build_linearmodel_features(catCols, contCols)

	classifier = LogisticRegression(labelCol = adjustedIndexerModel.getOutputCol(), featuresCol = features[-1].getOutputCol())

	pipeline = Pipeline(stages = [adjustedIndexer] + features + [classifier])

	build_classification(audit_df, pipeline, adjustedIndexerModel, classifier.getPredictionCol(), classifier.getProbabilityCol(), name)

if "Audit" in datasets:
	audit_df = load_csv("AuditNA")
	print(audit_df.dtypes)

	audit_df = cast_col(audit_df, "Age", DoubleType())
	audit_df = cast_col(audit_df, "Income", DoubleType())
	audit_df = cast_col(audit_df, "Hours", DoubleType())
	print(audit_df.dtypes)

	store_schema(audit_df, "AuditNA")

	build_classification_auditna(audit_df, "LogisticRegressionAuditNA")

def build_formula_auto(auto_df, name):
	sqlTransformer = SQLTransformer(statement = "SELECT *, CONCAT(origin, \"/\", cylinders) AS origin_cylinders FROM __THIS__")

	rFormula = RFormula(formula = "mpg ~ .", labelCol = "mpg", featuresCol = "featureVector")

	lr = LinearRegression(solver = "l-bfgs", labelCol = rFormula.getLabelCol(), featuresCol = rFormula.getFeaturesCol())

	build_regression(auto_df, Pipeline(stages = [sqlTransformer, rFormula, lr]), "mpg", lr.getPredictionCol(), name)

def build_modelchain_auto(auto_df, name):
	accelerationFormula = RFormula(formula = "acceleration ~ cylinders + displacement + horsepower + weight + cylinders:displacement:horsepower:weight", labelCol = "accelerationLabel", featuresCol = "accelerationFeatureVector")

	accelerationLr = LinearRegression(solver = "l-bfgs", labelCol = accelerationFormula.getLabelCol(), featuresCol = accelerationFormula.getFeaturesCol(), predictionCol = "predicted_acceleration")

	mpgFormula = RFormula(formula = "mpg ~ predicted_acceleration + model_year + origin + model_year:origin", labelCol = "mpg", featuresCol = "mpgFeatureVector")

	mpgLr = LinearRegression(solver = "l-bfgs", labelCol = mpgFormula.getLabelCol(), featuresCol = mpgFormula.getFeaturesCol())

	build_regression(auto_df, Pipeline(stages = [accelerationFormula, accelerationLr, mpgFormula, mpgLr]), "mpg", mpgLr.getPredictionCol(), name)

def build_regression_auto(auto_df, regressor, name):
	stages = []
	featureCols = []

	accelerationVectorAssembler = VectorAssembler(inputCols = ["acceleration"], outputCol = "accelerationVector")
	accelerationScaler = MaxAbsScaler(inputCol = accelerationVectorAssembler.getOutputCol(), outputCol = "scaledAcceleration")

	stages += [accelerationVectorAssembler, accelerationScaler]
	featureCols.append(accelerationScaler.getOutputCol())

	horsepowerWeightDiscretizer = QuantileDiscretizer(numBuckets = 10, inputCols = ["horsepower", "weight"], outputCols = ["discretizedHorsepower", "discretizedWeight"])

	stages.append(horsepowerWeightDiscretizer)
	featureCols += horsepowerWeightDiscretizer.getOutputCols()

	displacementBucketizer = Bucketizer(splits = [0, 100, 200, 300, 400, 500], inputCol = "displacement", outputCol = "bucketizedDisplacement")

	stages.append(displacementBucketizer)
	featureCols.append(displacementBucketizer.getOutputCol())

	weightVectorAssembler = VectorAssembler(inputCols = ["weight"], outputCol = "weightVector")
	weightScaler = MinMaxScaler(inputCol = weightVectorAssembler.getOutputCol(), outputCol = "scaledWeight")

	stages += [weightVectorAssembler, weightScaler]
	featureCols.append(weightScaler.getOutputCol())

	horsepowerDisplacementWeightInteraction = Interaction(inputCols = [horsepowerWeightDiscretizer.getOutputCols()[0], displacementBucketizer.getOutputCol(), "weight"], outputCol = "horsepowerBucket:displacementBucket:weight")

	stages.append(horsepowerDisplacementWeightInteraction)
	featureCols.append(horsepowerDisplacementWeightInteraction.getOutputCol())

	stringCols = ["cylinders", "model_year", "origin"]
	for stringCol in stringCols:
		stringIndexer = StringIndexer(inputCol = stringCol, outputCol = stringCol.lower() + "Index")
		stringIndexerModel = stringIndexer.fit(auto_df)

		ohe = OneHotEncoder(dropLast = False, inputCol = stringIndexerModel.getOutputCol(), outputCol = stringCol.lower() + "ClassVector")

		vectorSizeHint = VectorSizeHint(size = len(stringIndexerModel.labels), inputCol = ohe.getOutputCol())

		stages += [stringIndexerModel, ohe, vectorSizeHint]
		featureCols.append(ohe.getOutputCol())

	vectorAssembler = VectorAssembler(inputCols = featureCols, outputCol = "featureVector")

	regressor = regressor.setLabelCol("mpg").setFeaturesCol(vectorAssembler.getOutputCol())

	stages += [vectorAssembler, regressor]

	pipeline = Pipeline(stages = stages)

	if isinstance(regressor, DecisionTreeRegressor):
		paramGrid = ParamGridBuilder() \
			.addGrid(regressor.maxBins, [16, 32, 64]) \
			.addGrid(regressor.maxDepth, [4, 5, 6]) \
			.addGrid(regressor.minInstancesPerNode, [5, 10, 20]) \
			.build()

		regressionEvaluator = RegressionEvaluator(labelCol = regressor.getLabelCol(), predictionCol = regressor.getPredictionCol())

		crossValidator = CrossValidator(estimator = pipeline, estimatorParamMaps = paramGrid, evaluator = regressionEvaluator, numFolds = 3, seed = 63317)

		pipeline = Pipeline(stages = [crossValidator])

	build_regression(auto_df, pipeline, "mpg", regressor.getPredictionCol(), name)

def build_isotonic_auto(auto_df, regressor, name):
	vectorAssembler = VectorAssembler(inputCols = ["acceleration", "weight"], outputCol = "featureVector")

	regressor = regressor.setLabelCol("mpg").setFeaturesCol(vectorAssembler.getOutputCol())

	pipeline = Pipeline(stages = [vectorAssembler, regressor])

	build_regression(auto_df, pipeline, "mpg", regressor.getPredictionCol(), name)

if "Auto" in datasets:
	auto_df = load_csv("Auto")
	print(auto_df.dtypes)

	auto_df = cast_col(auto_df, "cylinders", StringType())
	auto_df = cast_col(auto_df, "model_year", StringType())
	auto_df = cast_col(auto_df, "origin", StringType())
	print(auto_df.dtypes)

	store_schema(auto_df, "Auto")

	build_formula_auto(auto_df, "LinearRegressionAuto")
	build_modelchain_auto(auto_df, "ModelChainAuto")

	build_regression_auto(auto_df, DecisionTreeRegressor(), "DecisionTreeAuto")
	build_regression_auto(auto_df, GBTRegressor(), "GBTAuto")
	build_regression_auto(auto_df, GeneralizedLinearRegression(family = "gaussian", link = "identity"), "GLMAuto")
	build_regression_auto(auto_df, RandomForestRegressor(numTrees = 13), "RandomForestAuto")

	build_isotonic_auto(auto_df, IsotonicRegression(isotonic = True, featureIndex = 0), "IsotonicRegressionIncrAuto")
	build_isotonic_auto(auto_df, IsotonicRegression(isotonic = False, featureIndex = 1), "IsotonicRegressionDecrAuto")

def build_regression_autona(df, name):
	catCols = ["cylinders", "model_year", "origin"]
	contCols = ["acceleration", "displacement", "horsepower", "weight"]

	features = build_linearmodel_features(catCols, contCols)

	regressor = LinearRegression(labelCol = "mpg", featuresCol = features[-1].getOutputCol())

	pipeline = Pipeline(stages = features + [regressor])

	build_regression(auto_df, pipeline, "mpg", regressor.getPredictionCol(), name)

if "Auto" in datasets:
	auto_df = load_csv("AutoNA")
	print(auto_df.dtypes)

	auto_df = cast_col(auto_df, "mpg", DoubleType())

	catCols = ["cylinders", "model_year", "origin"]
	contCols = ["acceleration", "displacement", "horsepower", "weight"]
	for catCol in catCols:
		auto_df = cast_col(auto_df, catCol, StringType())	
	for contCol in contCols:
		auto_df = cast_col(auto_df, contCol, DoubleType())
	print(auto_df.dtypes)

	store_schema(auto_df, "AutoNA")

	build_regression_autona(auto_df, "LinearRegressionAutoNA")

def build_regression_housing(husing_df, regressor, name):
	stages = []
	featureCols = []

	catCols = ["CHAS", "RAD", "TAX"]
	contCols = ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "PTRATIO", "B", "LSTAT"]

	featureCols += catCols

	ageImputer = Imputer(strategy = "median", missingValue = -1, inputCol = "AGE", outputCol = "imputedAge")
	
	stages.append(ageImputer)
	featureCols.append(ageImputer.getOutputCol())

	contCols.remove("AGE")

	contColImputer = Imputer(strategy = "mean", missingValue = -1.0, inputCols = contCols, outputCols = ["imputed" + contCol for contCol in contCols])

	stages.append(contColImputer)
	featureCols += contColImputer.getOutputCols()

	vectorAssembler = VectorAssembler(inputCols = featureCols, outputCol = "featureVector")

	vectorIndexer = VectorIndexer(inputCol = vectorAssembler.getOutputCol(), outputCol = "catFeatureVector")

	regressor = regressor.setLabelCol("MEDV").setFeaturesCol(vectorIndexer.getOutputCol())

	stages += [vectorAssembler, vectorIndexer, regressor]

	build_regression(housing_df, Pipeline(stages = stages), "MEDV", regressor.getPredictionCol(), name)

if "Housing" in datasets:
	housing_df = load_csv("Housing")
	print(housing_df.dtypes)

	store_schema(housing_df, "Housing")

	build_regression_housing(housing_df, DecisionTreeRegressor(), "DecisionTreeHousing")
	build_regression_housing(housing_df, GeneralizedLinearRegression(family = "gaussian", link = "identity", fitIntercept = False), "GLMHousing")
	build_regression_housing(housing_df, LinearRegression(solver = "l-bfgs"), "LinearRegressionHousing")
	build_regression_housing(housing_df, RandomForestRegressor(numTrees = 13), "RandomForestHousing")

def build_clustering_iris(iris_df, name):
	vectorAssembler = VectorAssembler(inputCols = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"], outputCol = "featureVector")

	kmeans = KMeans(k = 11, featuresCol = vectorAssembler.getOutputCol(), predictionCol = "cluster")

	build_clustering(iris_df, Pipeline(stages = [vectorAssembler, kmeans]), kmeans.getPredictionCol(), name)

def build_formula_iris(iris_df, name):
	speciesIndexer = StringIndexer(inputCol = "Species", outputCol = "speciesIndex")
	speciesIndexerModel = speciesIndexer.fit(iris_df)

	rFormula = RFormula(formula = "Species ~ Sepal_Length + Sepal_Width + Petal_Length + Petal_Width")

	lr = LogisticRegression(labelCol = rFormula.getLabelCol(), featuresCol = rFormula.getFeaturesCol())

	build_classification(iris_df, Pipeline(stages = [speciesIndexer, rFormula, lr]), speciesIndexerModel, lr.getPredictionCol(), lr.getProbabilityCol(), name)

def build_modelchain_iris(iris_df, name):
	speciesIndexer = StringIndexer(inputCol = "Species", outputCol = "speciesIndex")
	speciesIndexerModel = speciesIndexer.fit(iris_df)

	kmeansVectorAssembler = VectorAssembler(inputCols = ["Sepal_Length", "Petal_Length"], outputCol = "kmeansFeatureVector")

	kmeans = KMeans(k = 6, featuresCol = kmeansVectorAssembler.getOutputCol(), predictionCol = "cluster")

	lrVectorAssembler = VectorAssembler(inputCols = [kmeans.getPredictionCol()] + ["Sepal_Width", "Petal_Width"], outputCol = "lrFeatureVector")

	lr = LogisticRegression(labelCol = speciesIndexer.getOutputCol(), featuresCol = lrVectorAssembler.getOutputCol())

	build_classification(iris_df, Pipeline(stages = [speciesIndexer, kmeansVectorAssembler, kmeans, lrVectorAssembler, lr]), speciesIndexerModel, lr.getPredictionCol(), lr.getProbabilityCol(), name)

def build_classification_iris(iris_df, classifier, name):
	stages = []

	speciesIndexer = StringIndexer(inputCol = "Species", outputCol = "speciesIndex")
	speciesIndexerModel = speciesIndexer.fit(iris_df)

	vectorAssembler = VectorAssembler(inputCols = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"], outputCol = "featureVector")

	stages += [speciesIndexer, vectorAssembler]

	if isinstance(classifier, NaiveBayes):
		pass
	elif isinstance(classifier, MultilayerPerceptronClassifier):
		pca = PCA(k = 2, inputCol = vectorAssembler.getOutputCol(), outputCol = "pcaVector")
		
		stages.append(pca)
	else:
		standardScaler = StandardScaler(withMean = True, withStd = True, inputCol = vectorAssembler.getOutputCol(), outputCol = "scaledFeatureVector")

		stages.append(standardScaler)

	classifier = classifier.setLabelCol(speciesIndexer.getOutputCol()).setFeaturesCol(stages[-1].getOutputCol())

	stages.append(classifier)

	build_classification(iris_df, Pipeline(stages = stages), speciesIndexerModel, classifier.getPredictionCol(), classifier.getProbabilityCol(), name)

if "Iris" in datasets:
	iris_df = load_csv("Iris")
	print(iris_df.dtypes)

	store_schema(iris_df, "Iris")

	build_clustering_iris(iris_df, "KMeansIris")
	build_formula_iris(iris_df, "LogisticRegressionIris")
	build_modelchain_iris(iris_df, "ModelChainIris")

	build_classification_iris(iris_df, DecisionTreeClassifier(minInstancesPerNode = 3), "DecisionTreeIris")
	build_classification_iris(iris_df, NaiveBayes(), "NaiveBayesIris")
	build_classification_iris(iris_df, MultilayerPerceptronClassifier(layers = [2, 4, 3], seed = 13), "NeuralNetworkIris")
	build_classification_iris(iris_df, RandomForestClassifier(numTrees = 13, minInstancesPerNode = 20), "RandomForestIris")

def build_classification_sentiment(sentiment_df, classifier, name):
	stages = []

	scoreIndexer = StringIndexer(inputCol = "Score", outputCol = "scoreIndex")
	scoreIndexerModel = scoreIndexer.fit(sentiment_df)

	regexTokenizer = RegexTokenizer(pattern = "\\W+", inputCol = "Sentence", outputCol = "tokenizedSentence")

	stopWords = [str(i) for i in range(0, 10)]
	stopWords += StopWordsRemover.loadDefaultStopWords("english")

	stopWordsRemover = StopWordsRemover(caseSensitive = False, stopWords = stopWords, inputCol = regexTokenizer.getOutputCol(), outputCol = "filteredTokenizedSentence")

	stages += [scoreIndexer, regexTokenizer, stopWordsRemover]

	if isinstance(classifier, RandomForestClassifier):
		nMax = 3
	else:
		nMax = 2

	countCols = []

	for n in range(nMax):
		n += 1

		if n > 1:
			ngram = NGram(n = n, inputCol = stopWordsRemover.getOutputCol(), outputCol = "ngramSentence_1to" + str(n))

			stages.append(ngram)

		if isinstance(classifier, LinearSVC):
			vocabSize = 150
		elif isinstance(classifier, RandomForestClassifier):
			vocabSize = 500
		else:
			vocabSize = 50

		countVectorizer = CountVectorizer(binary = False, vocabSize = vocabSize, inputCol = stages[-1].getOutputCol(), outputCol = "countVector_1to" + str(n))

		stages.append(countVectorizer)
		countCols.append(countVectorizer.getOutputCol())

	vectorAssembler = VectorAssembler(inputCols = countCols, outputCol = "featureVector")

	stages.append(vectorAssembler)

	if isinstance(classifier, RandomForestClassifier):
		pass
	else:
		idf = IDF(inputCol = vectorAssembler.getOutputCol(), outputCol = "idfFeatureVector")

		stages.append(idf)

	classifier = classifier.setLabelCol(scoreIndexer.getOutputCol()).setFeaturesCol(stages[-1].getOutputCol())

	stages.append(classifier)

	if isinstance(classifier, GeneralizedLinearRegression):
		predictionCol = None
		probabilityCol = classifier.getPredictionCol()
	elif isinstance(classifier, LinearSVC):
		predictionCol = classifier.getPredictionCol()
		probabilityCol = None
	else:
		predictionCol = classifier.getPredictionCol()
		probabilityCol = classifier.getProbabilityCol()

	build_classification(sentiment_df, Pipeline(stages = stages), scoreIndexerModel, predictionCol, probabilityCol, name)

if "Sentiment" in datasets:
	sentiment_df = load_csv("Sentiment")
	print(sentiment_df.dtypes)

	store_schema(sentiment_df, "Sentiment")

	build_classification_sentiment(sentiment_df, DecisionTreeClassifier(maxDepth = 12, minInstancesPerNode = 3), "DecisionTreeSentiment")
	build_classification_sentiment(sentiment_df, GeneralizedLinearRegression(family = "binomial", link = "probit"), "GLMSentiment")
	build_classification_sentiment(sentiment_df, LinearSVC(threshold = 1.25), "LinearSVCSentiment")
	build_classification_sentiment(sentiment_df, RandomForestClassifier(numTrees = 17, minInstancesPerNode = 10), "RandomForestSentiment")

def build_associationrules_shopping(shopping_df, name):
	fpGrowth = FPGrowth(minConfidence = 0.01, minSupport = 0.01)

	build_associationrules(shopping_df, Pipeline(stages = [fpGrowth]), name)

if "Shopping" in datasets:
	shopping_df = load_csv("Shopping")
	print(shopping_df.dtypes)

	shopping_df = shopping_df.groupBy(col("transaction")).agg(collect_set("item").alias("items"))
	print(shopping_df.dtypes)

	store_schema(shopping_df, "Shopping")

	build_associationrules_shopping(shopping_df, "FPGrowthShopping")

def build_formula_visit(visit_df, name):
	sqlTransformer = SQLTransformer(statement = "SELECT *, IF(female == 1, 'M', 'F') AS gender, IF(married == 1, 0, 1) FROM __THIS__")

	rFormula = RFormula(formula = "docvis ~ . - female - married", labelCol = "docvis", featuresCol = "featureVector")

	glm = GeneralizedLinearRegression(family = "poisson", link = "sqrt", labelCol = rFormula.getLabelCol(), featuresCol = rFormula.getFeaturesCol())

	build_regression(visit_df, Pipeline(stages = [sqlTransformer, rFormula, glm]), "docvis", glm.getPredictionCol(), name)

if "Visit" in datasets:
	visit_df = load_csv("Visit")
	print(visit_df.dtypes)

	store_schema(visit_df, "Visit")

	build_formula_visit(visit_df, "GLMVisit")