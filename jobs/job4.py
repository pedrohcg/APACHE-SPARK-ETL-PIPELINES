from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, hour
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder \
    .appName("Pipeline ETL para Detectar Anomalias Financeiras - Job 4") \
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

# remove a coluna de features e scaledFeatures
df_resultado_clean = df_resultado.drop('features', 'scaledFeatures')
df_resultado_clean.show()

print(f"Total de linhas no dataframe: {df_resultado_clean.count()}")

df_resultado_clean.write.mode('overwrite').csv('data/resultado_job4', header=True)

spark.stop()

print("\nExecucao finalizada\n")