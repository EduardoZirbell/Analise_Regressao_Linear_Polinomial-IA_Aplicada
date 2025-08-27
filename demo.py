import numpy as np
import matplotlib.pyplot as plt

def lerDataSet():
    dados = {}
    
    with open('datasetFase1.txt', 'r') as arquivo:
        linhas = arquivo.readlines()
    
    for linha in linhas:
        linha = linha.strip()
        if '=' in linha and '[' in linha:
            # Separar nome da variável dos valores
            nome = linha.split('=')[0].strip()
            # Extrair os valores entre colchetes
            valores_str = linha.split('[')[1].split(']')[0]
            # Converter para lista de números
            valores = [float(v) for v in valores_str.split(';')]
            dados[nome] = valores
    return dados

def correlacao(x, y):
    n = len(x)
    media_x = sum(x) / n  # x̄
    media_y = sum(y) / n  # ȳ

    # Fórmula: r = Σ(x - x̄)(y - ȳ) / sqrt[Σ(x - x̄)²] * [Σ(y - ȳ)²]
    numerador = sum((xi - media_x) * (yi - media_y) for xi, yi in zip(x, y))
    denominador_x = sum((xi - media_x) ** 2 for xi in x)
    denominador_y = sum((yi - media_y) ** 2 for yi in y)
    r = numerador / np.sqrt(denominador_x * denominador_y)
    return r

def regressao(b0, b1, x):
    y = b1 + (b0 * x)
    return y

def getB0(x, y, b1):
    n = len(x)
    # Formúla: b0 = ȳ - b1*x̄
    media_x = sum(x) / n  # x̄
    media_y = sum(y) / n  # ȳ
    
    b0 = media_y - b1 * media_x
    return b0

def getB1(x, y):
    n = len(x)
    media_x = sum(x) / n  # x̄
    media_y = sum(y) / n  # ȳ

    # Fórmula: b1 = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
    numerador = sum((xi - media_x) * (yi - media_y) for xi, yi in zip(x, y))
    denominador = sum((xi - media_x) ** 2 for xi in x)

    b1 = numerador / denominador
    return b1

if __name__ == "__main__":
    dados = lerDataSet()
    
    def plot_dataset(x, y, b0, b1, r, title):
        info = format_regressao_info(b0, b1)
        plt.scatter(x, y, label='Dados')
        plt.plot(x, [b0 + b1 * xi for xi in x], color='red', label='Regressão Linear')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{title}\nr = {r:.2f}\n{info}')
        plt.legend()
        plt.show()

    def format_regressao_info(b0, b1):
        return f'beta0 = {b0:.4f}, beta1 = {b1:.5f}'

    for i in range(1, 4):
        x_key = f'x{i}'
        y_key = f'y{i}'
        x = dados[x_key]
        y = dados[y_key]
        
        b1 = getB1(x, y)
        b0 = getB0(x, y, b1)
        
        r = correlacao(x, y)
        
        plot_dataset(x, y, b0, b1, r, f'Dataset {i}')

    print("Análise dos DataSets:")
    print("DataSet 1 pode ser resolvido com uma regressão linear simples")
    print("DataSet 2 não pode ser resolvido porque necessita de uma regressão polinomial, pois apresenta uma relação não-linear entre as variáveis.")
    print("DataSet 3 apresenta dados discrepantes que impedem a aplicação de regressão linear simples, pois a maioria dos pontos está concentrada em uma única coordenada x.")