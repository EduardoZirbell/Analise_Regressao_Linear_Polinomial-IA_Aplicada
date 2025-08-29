import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

# Removido import do sklearn.linear_model.LinearRegression pois não está sendo usado

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv')
df = pd.read_csv(file_path, header=None)

df.columns = ["Tamanho", "Quartos", "Preco"]

print(df.head())  

media_preco = df["Preco"].mean()
menor_casa = df["Tamanho"].min()
quartos_casa_mais_cara = df.loc[df["Preco"].idxmax(), "Quartos"]

print("Média de preço das casas:", media_preco)
print("Menor casa (m²):", menor_casa)
print("Número de quartos da casa mais cara:", quartos_casa_mais_cara)

# ====== 4. Criar X e y ======
X = df[["Tamanho", "Quartos"]].values
y = df["Preco"].values

# ====== 5. Correlação ======
corr_tamanho_preco = df["Tamanho"].corr(df["Preco"])
corr_quartos_preco = df["Quartos"].corr(df["Preco"])
print("Correlação Tamanho vs Preço:", corr_tamanho_preco)
print("Correlação Quartos vs Preço:", corr_quartos_preco)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Gráfico de dispersão
ax.scatter(df["Tamanho"], df["Quartos"], df["Preco"], c='b', marker='o', alpha=0.7, label='Dados')

# Implementação manual da regressão linear múltipla
def fit_linear_regression(X, y):
    # Adiciona coluna de 1 para o intercepto
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Calcula os coeficientes usando a fórmula dos mínimos quadrados
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta_best

class LinearRegressionManual:
    def fit(self, X, y):
        self.coef_ = None
        self.intercept_ = None
        theta = fit_linear_regression(X, y)
        self.intercept_ = theta[0]
reg = LinearRegressionManual()
reg.fit(X, y)
print("Intercepto:", reg.intercept_)
print("Coeficientes:", reg.coef_)

reg = LinearRegressionManual()

# partir daqui é oq importa
theta = fit_linear_regression(X, y)

def prever_preco(tamanho, quartos, theta):
    return theta[0] + theta[1] * tamanho + theta[2] * quartos
# dado variavel para entender a H
tamanho_teste = 1650 
quartos_teste = 3    

preco_previsto = prever_preco(tamanho_teste, quartos_teste, theta)
print(f"Preço previsto para {tamanho_teste} m² e {quartos_teste} quartos: {preco_previsto:.2f}")

# colunas e título
ax.set_xlabel('Tamanho (m²)')
ax.set_ylabel('Quartos')
ax.set_zlabel('Preço')
plt.gcf().suptitle('Dispersão 3D: Tamanho, Quartos e Preço', fontsize=16, y=0.98)

# coeficientes de correlação no gráfico
corr_text = f"Corr(Tamanho, Preço): {corr_tamanho_preco:.2f}\nCorr(Quartos, Preço): {corr_quartos_preco:.2f}"
plt.gcf().text(0.15, 0.85, corr_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# linha de regressão
tamanho_range = np.linspace(df["Tamanho"].min(), df["Tamanho"].max(), 10)
quartos_range = np.linspace(df["Quartos"].min(), df["Quartos"].max(), 10)
Tamanho_grid, Quartos_grid = np.meshgrid(tamanho_range, quartos_range)
Preco_grid = theta[0] + theta[1] * Tamanho_grid + theta[2] * Quartos_grid
ax.plot_surface(Tamanho_grid, Quartos_grid, Preco_grid, color='r', alpha=0.4)


ax.scatter(tamanho_teste, quartos_teste, preco_previsto, c='r', marker='^', s=100, label='Teste')

plt.legend()
plt.show()