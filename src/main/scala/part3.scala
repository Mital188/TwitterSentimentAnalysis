import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.fpm.FPGrowth

object part3 {

  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      println("Insufficient parameters")
    }
    val sc = new SparkContext(new SparkConf().setAppName("tweetAnalysis"))

    val spark = SparkSession
      .builder()
      .appName("AirlineTweets_part3")
      .getOrCreate()

    val data = spark.read.option("header","true").option("inferSchema","true").csv(args(0))
    val transaction = data.groupBy("order_id").agg(collect_list("product_id") as "product")
    val fpgrowth = new FPGrowth().setItemsCol("product").setMinSupport(0.001).setMinConfidence(0.03)
    val model = fpgrowth.fit(transaction)

    val freqResult = model.freqItemsets.orderBy(desc("freq")).limit(10).take(10)
    val associationResult = model.associationRules.orderBy(desc("confidence")).limit(10).take(10)

    val frequency_result = sc.parallelize(freqResult)
    val association_result = sc.parallelize(associationResult)

    var output_freq = ""
    output_freq += "The top 10 frequent item sets are\t" + "\n"
    val freq = sc.parallelize(List(output_freq))
    var output_asso = ""
    output_asso += "The top 10 association rules are\t" + "\n"
    val association = sc.parallelize(List(output_asso))

    val final_solution = freq ++ frequency_result.map(_.mkString(",")) ++ association ++ association_result.map(_.mkString(","))
    final_solution.coalesce(1, true).saveAsTextFile(args(1))

  }
}
