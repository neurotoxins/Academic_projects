from pyspark import SparkContext, SparkConf

conf = SparkConf()
conf.setMaster("local[4]")
conf.setAppName("reduce")
conf.set("spark.executor.memory", "4g")

sc = SparkContext(conf=conf)

text = sc.wholeTextFiles("./sherlock.txt").map(lambda x: x[1].replace(',',' ').replace('.', ' ').replace('\n', ' ').lower());

words = text.map(lambda x:x.split()).flatMap(lambda x: [((x[i],x[i+1]),1) for i in range(0,len(x)-1)]);

common_words = words.reduceByKey(lambda x,y:x+y).map(lambda x:(x[1],x[0])).sortByKey(False);

print(common_words.take(10));

