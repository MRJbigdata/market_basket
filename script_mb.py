from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import concat_ws, rand, split, col, desc 

if __name__ == "__main__":

	conf = SparkConf()
	conf.setAppName("script_mb")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	df = sqlContext.read.load('hdfs://elephant:8020/user/labdata/order_products__prior.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')

	df_fp = df.groupBy(df.order_id).agg(F.collect_list(df.product_id).alias('product_id'))

	total = df_fp.count()

	p=0.001

	sampling = int(round(total*p))

	df_fp = df_fp.orderBy(rand()).limit(sampling)

	fpGrowth = FPGrowth(itemsCol="product_id", minSupport=0.0008, minConfidence=0.001)
	model = fpGrowth.fit(df_fp)


	df_freq = model.freqItemsets
	df_ar=model.associationRules


	df_supp = df_freq.withColumn("support", df_freq.freq/total)

	df_supp_ar = df_ar.join(df_supp,df_supp.items==df_ar.consequent, 'left').drop('items')

	df_lift = df_supp_ar.withColumn("lift", df_supp_ar.confidence/df_supp_ar.support)



	df_freq_new = df_freq.withColumn('items', concat_ws(',', 'items'))

	df_lift_new = df_lift.withColumn('antecedent', concat_ws(',', 'antecedent')).withColumn('consequent', concat_ws(',', 'consequent'))

	df_lift_new.show()

	df_lift_new.count()


	df_temp = df_lift_new.select('antecedent', F.split('antecedent', ',').alias('antecedent_split'),'consequent','confidence','freq','support','lift')

	df_temp = df_temp.select('antecedent',F.size('antecedent_split').alias('antecedent_length'),'antecedent_split','consequent','confidence','freq','support','lift')

	df_sizes = df_temp.select(F.size('antecedent_split').alias('antecedent_length'))

	df_max = df_sizes.agg(F.max('antecedent_length'))

	nb_columns = df_max.collect()[0][0]

	df_result = df_temp.select('antecedent','antecedent_length', *[df_temp['antecedent_split'][i] for i in range(nb_columns)],'consequent','confidence','freq','support','lift')

	df_result.show()


	############################################################# SALVAR .CSV
	#df_result.write.csv('hdfs://elephant:8020/user/labdata/df_result.csv', sep=',', header='true')
	#########################################################################

	df_result.drop('antecedent')

	df_result=df_result.select('antecedent_length','consequent','confidence','freq','support','lift','antecedent_split[0]','antecedent_split[1]','antecedent_split[2]','antecedent_split[3]')

	print(df_result.show(5))


	############################################################# SALVAR .CSV
	df_result.write.csv('hdfs://elephant:8020/user/labdata/teste/df_result.csv', sep=',')



