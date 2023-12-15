import os
import sqlite3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, hour
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import warnings
warnings.filterwarnings('ignore')

spark = SparkSession.builder \
    .appName("Pipeline ETL para Detectar Anomalias Financeiras - Job 5") \
    .getOrCreate()

spark.sparkContext.setLogLevel('ERROR')

df = spark.read.csv('data/dados1_cap05.csv', header=True, inferSchema=True)

#Cria novas colunas com os numeros do campo data
df = df.withColumn("Ano", year(df["Data"]))
df = df.withColumn("Mes", month(df["Data"]))
df = df.withColumn("Dia", dayofmonth(df["Data"]))
df = df.withColumn("Hora", hour(df["Data"]))

features = ['Valor', 'Ano', 'Mes', 'Dia', 'Hora']

#Transform serve para aplicar a transformacao ao dataframe nesse caso vai aplicar ao dataframe o dataframe resultante do assembler
#Pre requisito para usar machine learning
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)

#Normaliza os dados | pre requisito para o KMeans funcionar
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
df = scaler.fit(df).transform(df)

kmeans = KMeans(featuresCol='scaledFeatures', k=3)

#Treina o modelo KMeans
modelo = kmeans.fit(df)

# Aplica o modelo treinado ao dataframe
df_resultado = modelo.transform(df)

cluster_counts = df_resultado.groupBy('prediction').count()
cluster_counts.show()

# remove a coluna de features e scaledFeatures
df_resultado_clean = df_resultado.drop('features', 'scaledFeatures')
df_resultado_clean.show()

print(f"Total de linhas no dataframe: {df_resultado_clean.count()}")

pandas_df = df_resultado_clean.toPandas()

output_dir = 'data/resultado_job5'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

conn = sqlite3.connect("data/resultado_job5/database_ml.db")
pandas_df.to_sql("tb_clusters", conn, if_exists="replace", index=False)
conn.close()

spark.stop()

print("\nExecucao finalizada\n")