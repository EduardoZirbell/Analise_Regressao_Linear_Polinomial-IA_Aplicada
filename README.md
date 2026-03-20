# 📊 Projeto de Regressão e Análise de Dados

Este trabalho tem como objetivo aplicar, na prática, conceitos de estatística e aprendizado de máquina, com foco em regressão linear e análise de dados.

O projeto foi dividido em 3 fases, onde cada uma aborda um tipo diferente de análise e modelo.

---

## 🚀 Tecnologias Utilizadas

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## 📁 Estrutura do Projeto


├── fase1.ipynb
├── fase2.ipynb
├── fase3.ipynb
├── data.csv
├── data_preg.csv
└── README.md


---

## 📌 Fase 1 – Correlação e Regressão Linear

Nessa fase foi analisada a relação entre duas variáveis utilizando correlação e regressão linear.

### O que foi feito:

- Cálculo do coeficiente de correlação  
- Cálculo dos coeficientes da regressão (β₀ e β₁)  
- Gráficos de dispersão  
- Plot da reta de regressão  

### Observações:

- Um dos datasets apresentou uma boa relação linear  
- Outro não seguiu um padrão linear claro  
- Houve presença de outliers em um dos casos, o que prejudicou a análise  

---

## 📌 Fase 2 – Regressão Linear Múltipla

Aqui foi trabalhado com mais de uma variável para prever um resultado (preço de casas).

### Dados utilizados:

- Tamanho da casa  
- Número de quartos  
- Preço  

### O que foi feito:

- Análise estatística dos dados  
- Cálculo de correlação entre variáveis  
- Regressão linear múltipla (manual e com biblioteca)  
- Gráfico 3D  
- Predição de valores  

### Observações:

- O tamanho da casa teve maior impacto no preço  
- O número de quartos teve menor influência  
- Os resultados entre a implementação manual e o Scikit-learn foram bem próximos  

---

## 📌 Fase 3 – Regressão Polinomial e Overfitting

Nesta fase foi analisado o comportamento de modelos mais complexos.

### O que foi feito:

- Regressão polinomial (graus 1, 2, 3 e 8)  
- Cálculo do erro quadrático médio (EQM)  
- Separação entre treino e teste  
- Avaliação com R²  

### Observações:

- Modelos mais complexos se ajustam melhor aos dados de treino  
- Porém podem causar overfitting  
- O melhor modelo é aquele que tem bom desempenho nos dados de teste  

---

## ⚠️ Conceitos Importantes

- **Overfitting:** modelo se ajusta demais aos dados de treino  
- **Underfitting:** modelo simples demais  
- **EQM:** mede o erro das previsões  
- **R²:** mede o quanto o modelo explica os dados  

---

## ▶️ Como Executar

1. Instale as dependências:


pip install numpy pandas matplotlib scikit-learn


2. Execute o Jupyter Notebook:


jupyter notebook


3. Abra os arquivos:

- `fase1.ipynb`  
- `fase2.ipynb`  
- `fase3.ipynb`  

---

## 📚 Conclusão

O projeto permitiu entender melhor como funcionam os modelos de regressão e como aplicá-los na prática.

Também mostrou a importância da análise dos dados e da validação com dados de teste, evitando problemas como overfitting.

---
