# Analise_Correlacao-Regressao_Linear

📊 Projeto de Regressão e Análise de Dados: 

Este trabalho tem como objetivo aplicar, na prática, alguns conceitos importantes de estatística e aprendizado de máquina, principalmente envolvendo regressão linear e análise de dados.
A ideia geral foi pegar diferentes conjuntos de dados e analisar como as variáveis se relacionam entre si, além de tentar prever valores com base nessas relações.
O projeto foi dividido em 3 fases, cada uma com um foco diferente.

🚀 Tecnologias Utilizadas: 
Python
NumPy
Pandas
Matplotlib
Scikit-learn

📌 Fase 1 – Correlação e Regressão Linear: 

Nessa primeira parte, o foco foi entender a relação entre duas variáveis, usando conceitos básicos como correlação e regressão linear.
Foi calculado o coeficiente de correlação (r), que indica o quanto as variáveis estão relacionadas, e também foi feita a regressão linear para tentar prever valores.

O que foi feito: 
Cálculo da correlação
Cálculo dos coeficientes da reta (β₀ e β₁)
Gráficos de dispersão
Plot da reta de regressão

Observações: 
Um dos datasets apresentou uma boa relação linear
Outro já não seguia um padrão linear muito claro
Teve também um caso com muitos outliers, o que prejudicou bastante a análise

📌 Fase 2 – Regressão Linear Múltipla: 
Aqui o objetivo foi trabalhar com mais de uma variável explicativa, analisando como elas influenciam em um resultado (no caso, preço de casas).

Dados utilizados: 
Tamanho da casa
Número de quartos
Preço

O que foi feito: 
Análise estatística dos dados
Verificação de correlação entre variáveis
Implementação da regressão múltipla (manual e com biblioteca)
Geração de gráfico 3D
Realização de previsões

Observações: 
O tamanho da casa teve um impacto bem maior no preço
Já o número de quartos não influenciou tanto quanto esperado
Os resultados entre a implementação própria e o Scikit-learn ficaram bem próximos

📌 Fase 3 – Regressão Polinomial e Overfitting: 
Nessa fase foi estudado como modelos mais complexos se comportam, principalmente em relação ao overfitting.
Foram testados modelos com diferentes graus (1, 2, 3 e 8), comparando os resultados.

O que foi feito: 
Regressão polinomial
Cálculo do erro quadrático médio (EQM)
Separação entre treino e teste
Avaliação usando R²

Observações: 
Modelos mais complexos tendem a se ajustar melhor aos dados de treino
Porém, nem sempre isso significa que são melhores
Em alguns casos ocorreu overfitting, onde o modelo "decorou" os dados

⚠️ Conceitos Importantes: 
Overfitting: quando o modelo se ajusta demais aos dados de treino
Underfitting: quando o modelo é simples demais
EQM: mede o erro das previsões
R²: indica o quanto o modelo consegue explicar os dados

▶️ Como Executar: 
Instalar as bibliotecas necessárias:
pip install numpy pandas matplotlib scikit-learn

Rodar o Jupyter: 
jupyter notebook

Abrir os arquivos: 
fase1.ipynb
fase2.ipynb
fase3.ipynb

📚 Conclusão: 
De forma geral, o trabalho ajudou a entender melhor como funcionam os modelos de regressão e como eles podem ser aplicados em problemas reais.
Também foi possível perceber que nem sempre um modelo mais complexo é melhor, e que a análise dos dados é uma parte muito importante do processo.
Além disso, ficou claro que a validação com dados de teste é essencial pra evitar conclusões erradas.
