import os
import sqlite3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
import pandas as pd

spark = SparkSession.builder \
    .appName("Pipeline ETL para Detectar Anomalias Financeiras - Job 1") \
    .getOrCreate()

df = spark.read.csv('data/dados1_cap05.csv', header=True, inferSchema=True)

spark.sparkContext.setLogLevel('ERROR')

df_dados_agregados = df.groupBy("ID_Cliente").agg(avg("Valor").alias("Media_Transacao"))

pandas_df = df_dados_agregados.toPandas()

output_dir = 'data/resultado_job3'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
conn = sqlite3.connect("data/resultado_job3/database.db")
pandas_df.to_sql("tb_anomalias_agregadas", conn, if_exists="replace", index=False)
conn.close()

spark.stop()

print("Execucao finalizada")