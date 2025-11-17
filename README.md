# 🎧 Sistema de Recomendação de Músicas com PySpark

Filtragem Colaborativa com ALS (Alternating Least Squares)

Este projeto implementa um sistema de recomendação musical usando PySpark e o algoritmo ALS, técnica amplamente utilizada em plataformas como Spotify e Deezer para prever quais músicas um usuário provavelmente vai gostar.
---
# 🎯 Objetivo

Criar um sistema de recomendação escalável baseado em filtragem colaborativa, capaz de prever notas de músicas e gerar recomendações personalizadas para cada usuário.
---
| Coluna    | Tipo  | Descrição     |
| --------- | ----- | ------------- |
| `userId`  | int   | ID do usuário |
| `trackId` | int   | ID da música  |
| `rating`  | float | Nota (0 a 5)  |

---
# ⚙️ Tecnologias Utilizadas

Python 3

PySpark (MLlib)

ALS (Alternating Least Squares)

RMSE (Root Mean Squared Error)

Google Colab
---
# 🛠️ Etapas do Projeto
## 1. Carregamento e Preparação dos Dados
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("RecomendacaoMusica").getOrCreate()

df = spark.read.csv("/content/musicas.csv", header=True, inferSchema=True)

df = df.select(
    col("userId").cast("int"),
    col("trackId").cast("int"),
    col("rating").cast("float")
)


## 2. Divisão entre Treino e Teste
treino, teste = df.randomSplit([0.8, 0.2], seed=42)

## 3. Criação e Treinamento do Modelo (ALS)
from pyspark.ml.recommendation import ALS

als = ALS(
    maxIter=10,
    regParam=0.1,
    rank=10,
    userCol="userId",
    itemCol="trackId",
    ratingCol="rating",
    coldStartStrategy="drop"
)

modelo = als.fit(treino)


modelo = als.fit(treino)

## 4. Avaliação do Modelo (RMSE)
from pyspark.ml.evaluation import RegressionEvaluator

predicoes = modelo.transform(teste)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predicoes)
print(f"RMSE: {rmse:.2f}")
---
# 📊 Resultado:
| Métrica  | Valor    |
| -------- | -------- |
| **RMSE** | **1.21** |

## 5. Geração de Recomendações
🔝 Recomendando Top-5 músicas por usuário:
recomendacoes = modelo.recommendForAllUsers(5)
recomendacoes.show(truncate=False)
---
#📌 Exemplo de Saída (Formatação ilustrativa)
+------+-------------------------------------------------------------+
|userId|recommendations                                              |
+------+-------------------------------------------------------------+
|12    |[[44,4.91], [7,4.73], [33,4.70], [18,4.82], [62,4.59]]       |
|33    |[[18,4.82], [62,4.59], [90,4.51], [7,4.40], [55,4.32]]       |
+------+-------------------------------------------------------------+

---
# 📊 Possíveis Melhorias Futuras

Tunagem com CrossValidator

Métricas de ranking (MAP, NDCG, Precision@k)

Combinação de modelos (híbrido colaborativo + conteúdo)

Deploy via API com FastAPI + Spark

# 🧠 Principais Aprendizados

ALS captura padrões profundos entre usuários com preferências semelhantes.

O modelo escalou sem perda significativa de desempenho.

RMSE sólido para um modelo colaborativo sem features adicionais.
---
