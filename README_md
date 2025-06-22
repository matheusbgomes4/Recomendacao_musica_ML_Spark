# 🎧 Sistema de Recomendação de Músicas com PySpark

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matheusbgomes4/Recomendador_Musical_PySpark/blob/main/Recomendacao_musica.ipynb)

Este projeto implementa um sistema de recomendação musical baseado em **filtragem colaborativa** usando o algoritmo **ALS (Alternating Least Squares)** da biblioteca **PySpark MLlib**.  
O objetivo é prever quais músicas um usuário tem maior probabilidade de gostar, com base em seu histórico de avaliações.

---

## 🎯 Objetivo

Criar um modelo escalável de aprendizado de máquina que recomenda músicas personalizadas para cada usuário, a partir de dados de interações anteriores.  
O sistema simula o funcionamento de plataformas como **Spotify**, **Deezer** ou **YouTube Music**, mas em ambiente educacional com Spark.

---

## 📁 Estrutura dos Dados

| Coluna      | Tipo    | Descrição                                   |
|-------------|---------|---------------------------------------------|
| `userId`    | int     | Identificador único do usuário              |
| `trackId`   | int     | Identificador único da música               |
| `rating`    | float   | Nota dada pelo usuário à música (0 a 5)     |

> *As notas representam a preferência do usuário por determinada música.*

---

## ⚙️ Tecnologias Utilizadas

- 🐍 **Python 3**
- ⚡ **Apache Spark (PySpark MLlib)**
- 🔁 **ALS (Alternating Least Squares)**
- 📊 RMSE (Root Mean Squared Error)
- ☁️ **Google Colab** para execução online

---

## 🛠️ Etapas do Projeto

1. **Leitura e pré-processamento dos dados**
   - Conversão de tipos
   - Eliminação de registros inconsistentes

2. **Construção do modelo**
   - Treinamento com `ALS()`, com `implicitPrefs=False`
   - Divisão treino/teste (75% / 25%)

3. **Avaliação do modelo**
   - Cálculo do **RMSE** no conjunto de teste

4. **Geração das recomendações**
   - `recommendForAllUsers(numItems=5)`
   - Sugestão de top-5 músicas para cada usuário

---

## 📊 Resultados Obtidos

| Métrica | Resultado |
|---------|-----------|
| RMSE    | 1.21      |

> RMSE (Root Mean Squared Error) avalia a precisão do modelo.  
> Quanto menor o valor, melhor a qualidade da previsão.

---
