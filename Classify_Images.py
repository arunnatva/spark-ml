
### Reference: https://medium.com/linagora-engineering/making-image-classification-simple-with-spark-deep-learning-f654a8b876b8

### Fire up a pyspark session & add spark deep learning libraries to the classpath

export PYSPARK_PYTHON=/opt/anaconda3/bin/python3
export SPARK_HOME=/usr/hdp/current/spark2-client
$SPARK_HOME/bin/pyspark --packages databricks:spark-deep-learning:0.1.0-spark2.1-s_2.11 --master yarn --executor-memory 3g --driver-memory 5g --conf spark.yarn.executor.memoryOverhead=5120
 
### Add the spark deep-learning jars into the classpath

import sys,glob,os
sys.path.extend(glob.glob(os.path.join(os.path.expanduser("~"),".ivy2/jars/*.jar")))
 
### PySpark code to read images, create spark ml pipeline, train the mode & predict

from sparkdl import readImages
from pyspark.sql.functions import lit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
 
img_dir = "/user/arun/TrainingData"
 
cat1_df = readImages(img_dir + "/cat1").withColumn("label", lit(1))
cat2_df = readImages(img_dir + "/cat2").withColumn("label", lit(2))
cat3_df = readImages(img_dir + "/cat3").withColumn("label", lit(3))
cat4_df = readImages(img_dir + "/cat4").withColumn("label", lit(4))
cat5_df = readImages(img_dir + "/cat5").withColumn("label", lit(5))
 
//Split the images where 90% of them go to training data, 10% go to test data
 
cat1_train, cat1_test = cat1_df.randomSplit([0.9, 0.1])
cat2_train, cat2_test = cat2_df.randomSplit([0.9, 0.1])
cat3_train, cat3_test = cat3_df.randomSplit([0.9, 0.1])
cat4_train, cat4_test = cat4_df.randomSplit([0.9, 0.1])
cat5_train, cat5_test = cat5_df.randomSplit([0.9, 0.1])
 
train_df = cat1_train.unionAll(cat2_train).unionAll(cat3_train).unionAll(cat4_train).unionAll(cat5_train)
test_df = cat1_test.unionAll(cat2_test).unionAll(cat3_test).unionAll(cat4_test).unionAll(cat5_test)
 
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
 
p = Pipeline(stages=[featurizer, rf])
p_model = p.fit(train_df)
 
predictions = p_model.transform(test_df)
predictions.select("filePath", "label", "prediction").show(200,truncate=False)
preds_vs_labels = predictions.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("accuracy of predictions by model = " + str(evaluator.evaluate(preds_vs_labels)))
 
# TRY TO CLASSIFY CAT 5 IMAGES, AND SEE HOW CLOSE THEY GET IN PREDICTING
cat5_imgs = readImages(img_dir + "/cat5").withColumn("label", lit(5))
pred5 = p_model.transform(cat5_imgs)
pred5.select("filePath","label","prediction").show(200,truncate=False)
