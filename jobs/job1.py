from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Pipeline ETL para Detectar Anomalias Financeiras - Job 1") \
    .getOrCreate()

df = spark.read.csv('data/dados1_cap05.csv', header=True, inferSchema=True)

spark.sparkContext.setLogLevel('ERROR')

df.show()

#Limite para considerar uma transacao como anomalia
LIMITE_ANOMALIA = 50000

#Filtra as anomalias
anomalias = df.filter(col('Valor') > LIMITE_ANOMALIA)

anomalias.show()

print(f"Total de linhas no DataFrame de anomalias: {anomalias.count()}")

anomalias.write.mode('overwrite').csv('data/resultado_job1', header=True)

spark.stop()

print("Execucao do job concluida com sucesso")