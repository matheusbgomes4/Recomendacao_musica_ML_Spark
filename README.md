# 🎧 Sistema de Recomendação de Músicas com PySpark

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matheusbgomes4/Recomendador_Musical_PySpark/blob/main/Recomendacao_musica.ipynb)

Este projeto apresenta a construção de um sistema de recomendação musical utilizando **filtragem colaborativa com o algoritmo ALS (Alternating Least Squares)** do PySpark.  
A proposta é prever quais músicas um usuário provavelmente vai gostar, com base em suas avaliações anteriores — um conceito aplicado em plataformas como Spotify e Deezer.

---

## 🎯 Objetivo

O objetivo deste projeto é desenvolver uma solução prática e escalável de recomendação de músicas, capaz de personalizar sugestões de acordo com os gostos dos usuários.  
O modelo identifica padrões de comportamento entre usuários com preferências semelhantes e utiliza esses padrões para prever futuras interações.

---

## 📁 Estrutura dos Dados

| Coluna    | Tipo    | Descrição                               |
|-----------|---------|------------------------------------------|
| `userId`  | int     | Identificador único do usuário           |
| `trackId` | int     | Identificador único da música            |
| `rating`  | float   | Nota dada pelo usuário à música (0 a 5)  |

---

## ⚙️ Tecnologias Utilizadas

- Python 3  
- PySpark (MLlib)  
- ALS (Alternating Least Squares)  
- Google Colab  
- RMSE (Root Mean Squared Error)

---

## 🛠️ Etapas do Projeto

1. **Preparação dos dados**
   - Leitura e limpeza
   - Conversão de tipos
2. **Criação do modelo**
   - Treinamento com ALS
   - Configuração de hiperparâmetros
   - Divisão em treino e teste
3. **Avaliação**
   - Cálculo do RMSE no conjunto de teste
4. **Geração de recomendações**
   - Top-N sugestões de músicas para cada usuário

---

## 📊 Resultado

| Métrica | Valor obtido |
|---------|--------------|
| RMSE    | 1.21         |

---
