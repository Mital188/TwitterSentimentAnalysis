import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.sql.Row


object part1 {

  var rdd: RDD[String] = null
  val sc = new SparkContext(new SparkConf().setAppName("tweetAnalysis_part1"))

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("tweetAnalysis_part1")
      .getOrCreate()

    val sqlContext = spark.sqlContext

    import spark.implicits._


    if (args.length != 2) {
      println("Usage: input output ")
    }

    var tweets= sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load(args(0))
    tweets = tweets.filter($"text".isNotNull)

    // Splits the sentence into words
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    //Words are converted into term-frequency vectors
    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    //Filters out the stop words
    val filteredWords = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("filteredWords")
    //converts string index into a numerical value
    val indexer = new StringIndexer().setInputCol("airline_sentiment").setOutputCol("label").fit(tweets)

    //Classifiers
    //Logistic Regression
    val lr = new LogisticRegression().setMaxIter(10).setFeaturesCol("features").setLabelCol("label")
    //Decision Tree classifier
    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")

    //Pipelines:
    // Logistic Regression Pipeline
    val pipeline_lr = new Pipeline().setStages(Array(tokenizer, filteredWords, hashingTF, indexer,lr))
    //Decision Tree Pipeline
    val pipeline_dt = new Pipeline().setStages(Array(tokenizer, filteredWords, hashingTF, indexer,dt))

    //Classification Models using ParamGrid
    //Logistic Regression Model
    val paramGrid_lr = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()
    //Decision Tree Model
    val paramGrid_dt = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(dt.maxDepth, Array(2,3,4,5))
      .build()
    //Evaluator
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")

    //CrossValidation -> tio find the best fit model
    //Logistic Regression
    val cv_lr = new CrossValidator()
      .setEstimator(pipeline_lr)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid_lr)
      .setNumFolds(3)
    //Decision Tree
    val cv_dt = new CrossValidator()
      .setEstimator(pipeline_dt)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid_dt)
      .setNumFolds(3)

    //Splitting the Data for training and testing
    val Array(train,test) = tweets.randomSplit(Array(0.8, 0.2))

    //Best fit model
    val cvModel_lr = cv_lr.fit(train)
    val cvModel_dt = cv_dt.fit(train)

    //Predicting Test Data
    val final_prediction_lr = cvModel_lr.transform(test)
    val final_prediction_dt = cvModel_dt.transform(test)

    var accuracy_lr = 0.0
    var precision_lr = 0.0
    var recall_lr = 0.0
    var f1score_lr = 0.0

    var accuracy_dt = 0.0
    var precision_dt = 0.0
    var recall_dt = 0.0
    var f1score_dt = 0.0

    // Calculate metrics for logistic Regression
    def displayMetrics_lr(values : RDD[(Double, Double)]) {
      val metrics = new MulticlassMetrics(values)
      accuracy_lr = metrics.accuracy
      precision_lr = metrics.weightedPrecision
      recall_lr = metrics.weightedRecall
      f1score_lr = metrics.weightedFMeasure
    }

    // Calculate metrics for Decision Tree
    def displayMetrics_dt(values : RDD[(Double, Double)]) {
      val metrics = new MulticlassMetrics(values)
      accuracy_dt = metrics.accuracy
      precision_dt = metrics.weightedPrecision
      recall_dt = metrics.weightedRecall
      f1score_dt = metrics.weightedFMeasure
    }

    val PredictionAndLabels_lr = final_prediction_lr.select("prediction", "label").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    displayMetrics_lr(PredictionAndLabels_lr)

    val PredictionAndLabels_dt = final_prediction_dt.select("prediction", "label").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    displayMetrics_dt(PredictionAndLabels_dt)


    var output = ""

    //Print metrics for both Evaluators
    output += "----Metrics for Decision Tree----\n"

    output += "Accuracy for Decision Tree: \t" + accuracy_dt + "\n"

    output += "Precision for Decision Tree: \t" + precision_dt + "\n"

    output += "Recall for Decision Tree: \t" + recall_dt + "\n"

    output += "F1Score for Decision Tree: \t" + f1score_dt + "\n"

    output += "----Metrics for Logistic Regression----\n"

    output += "Accuracy for Logistic Regression: \t" + accuracy_lr + "\n"

    output += "Precision for Logistic Regression: \t" + precision_lr + "\n"

    output += "Recall for Logistic Regression: \t" + recall_lr + "\n"

    output += "F1Score for Logistic Regression: \t" + f1score_lr + "\n"

    rdd = sc.parallelize(List(output))

    rdd.coalesce(1, true).saveAsTextFile(args(1))

  }
}
