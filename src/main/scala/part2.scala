import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.functions._
import scala.collection.mutable

object part2 {
  def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("topicModel"))
    val spark = SparkSession.builder().appName("AirlineTweets").getOrCreate()

    import spark.implicits._

    var tweets= spark.read.format("csv").option("header", "true").load(args(0))
    tweets = tweets.filter($"text".isNotNull)

    tweets = tweets.withColumn("airline_sentiment", when(col("airline_sentiment").equalTo("neutral"), 2.5).otherwise(col("airline_sentiment")))
    tweets = tweets.withColumn("airline_sentiment", when(col("airline_sentiment").equalTo("positive"), 5.0).otherwise(col("airline_sentiment")))
    tweets = tweets.withColumn("airline_sentiment", when(col("airline_sentiment").equalTo("negative"), 1.0).otherwise(col("airline_sentiment")))

    val avg_sentiment = tweets.groupBy("airline").agg(mean("airline_sentiment"))

    val airline_max = avg_sentiment.orderBy($"avg(airline_sentiment)".desc).select("airline").take(1).map(_.getString(0)).mkString(" ")
    val airline_min = avg_sentiment.orderBy($"avg(airline_sentiment)".asc).select("airline").take(1).map(_.getString(0)).mkString(" ")

    val stopWordSet = StopWordsRemover.loadDefaultStopWords("english").toSet

    var corpus_airline_max = tweets.filter(tweets("airline")=== airline_max).select("text").map(x=>x.toString()).rdd
    var corpus_airline_min = tweets.filter(tweets("airline")=== airline_min ).select("text").map(x=>x.toString()).rdd
//    val stopWordSet = StopWordsRemover.loadDefaultStopWords("english").toSet

    val tokenized_max: RDD[Seq[String]] = corpus_airline_max.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3).filter(token => !stopWordSet.contains(token)).filter(_.forall(java.lang.Character.isLetter)))
    val tokenized_min: RDD[Seq[String]] = corpus_airline_min.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3).filter(token => !stopWordSet.contains(token)).filter(_.forall(java.lang.Character.isLetter)))

    val termCounts_max: Array[(String, Long)] = tokenized_max.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
    val termCounts_min: Array[(String, Long)] = tokenized_min.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)

    val numStopwords = 20
    val vocabArray_max: Array[String] = termCounts_max.takeRight(termCounts_max.size - numStopwords).map(_._1)
    val vocabArray_min: Array[String] = termCounts_min.takeRight(termCounts_min.size - numStopwords).map(_._1)

    val vocab_max: Map[String, Int] = vocabArray_max.zipWithIndex.toMap
    val vocab_min: Map[String, Int] = vocabArray_min.zipWithIndex.toMap

    val documents_max: RDD[(Long, Vector)] =
      tokenized_max.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab_max.contains(term)) {
            val idx = vocab_max(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab_max.size, counts.toSeq))
      }
    val documents_min: RDD[(Long, Vector)] =
      tokenized_min.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab_min.contains(term)) {
            val idx = vocab_min(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab_min.size, counts.toSeq))
      }

    val numTopics = 10
    val lda = new LDA().setK(numTopics).setMaxIterations(10)
    val ldaModel_max = lda.run(documents_max)
    val ldaModel_min = lda.run(documents_min)

    val topicIndices_max = ldaModel_max.describeTopics(maxTermsPerTopic = 10)
    var output_max = ""
    topicIndices_max.foreach { case (terms, termWeights) =>
      output_max += "TOPIC:"
      terms.zip(termWeights).foreach { case (term, weight) =>
        output_max += {vocabArray_max(term.toInt)}
        output_max += "\t" + weight + "\n"
      }
      output_max += "\n\n"
    }

    val topicIndices_min = ldaModel_min.describeTopics(maxTermsPerTopic = 10)
    var output_min = ""
    topicIndices_min.foreach { case (terms, termWeights) =>
      output_min += "TOPIC:"
      terms.zip(termWeights).foreach { case (term, weight) =>
        output_min += {vocabArray_min(term.toInt)}
        output_min += "\t" + weight + "\n"
      }
      output_min += "\n\n"
    }

    sc.parallelize(List(output_max + "/n" + output_min)).saveAsTextFile(args(1))

  }
}
