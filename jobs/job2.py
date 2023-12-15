from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
import pandas as pd

spark = SparkSession.builder \
    .appName("Pipeline ETL para Detectar Anomalias Financeiras - Job 1") \
    .getOrCreate()

df = spark.read.csv('data/dados1_cap05.csv', header=True, inferSchema=True)

spark.sparkContext.setLogLevel('ERROR')

df_dados_agregados = df.groupBy("ID_Cliente").agg(avg("Valor").alias("Media_Transacao"))

print(f"\nTotal de linhas no Dataframe df_dados_agregados: {df_dados_agregados.count()}")

df_dados_agregados.write.mode('overwrite').parquet('data/resultado_job2')

spark.stop()

print("\nExecucao finalizada")