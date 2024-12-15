import pandas as pd
import numpy as np
import random


def generate_node_attributes(num_nodes):
    """Gera atributos fictícios para cada nó"""
    nodes = {}

    for node in range(num_nodes):
        # Status Realizado ou não (0 ou 1)
        status = random.choice([0, 1])

        # Dias desde última fiscalização (1-10 dias)
        dias = random.randint(1, 10)

        # Área de gramado (1-3, representando faixas de tamanho)
        area_manutencao = random.randint(1, 3)

        # Área nobre (0 ou 1)
        area_nobre = random.choice([0, 1])

        nodes[node] = {
            'dias': dias,
            'area_manutencao': area_manutencao,
            'area_nobre': area_nobre
            ,'status': status
        }

    return nodes


def calculate_new_cost(distance, node_i_attrs, node_j_attrs, weights):
    """Calcula o novo custo usando a função objetivo completa"""
    alpha, beta, gamma, delta = weights

    # Componente de distância
    cost = alpha * distance

    # Penalidade por dias sem fiscalização (para ambos os nós)
    cost += beta * max(0, node_i_attrs['dias'] - 7)
    cost += beta * max(0, node_j_attrs['dias'] - 7)

    # Componente de área de gramado (para ambos os nós)
    cost += gamma * (1 / node_i_attrs['area_manutencao'])
    cost += gamma * (1 / node_j_attrs['area_manutencao'])

    # Bonificação por área nobre (para ambos os nós)
    cost -= delta * node_i_attrs['area_nobre']
    cost -= delta * node_j_attrs['area_nobre']

    return cost


def process_matrix(df):
    """
    Processa a matriz de distâncias e gera nova matriz de custos
    Recebe um DataFrame pandas já formatado
    """
    # Definir pesos para cada componente
    weights = {
        'alpha': 0.2,  # Peso para distância
        'beta': 0.4,  # Peso para dias sem fiscalização
        'gamma': 0.1,  # Peso para área de gramado
        'delta': 0.3  # Peso para área nobre
    }

    # Gerar atributos fictícios para os nós
    nodes = generate_node_attributes(len(df))

    # Criar nova matriz com os mesmos índices e colunas
    new_df = df.copy()

    # Calcular novos custos
    for i in df.index:
        for j in df.columns:
            if not pd.isna(df.loc[i, j]):
                i_idx = df.index.get_loc(i)
                j_idx = df.columns.get_loc(j)

                if(nodes[i_idx]['status']==1):
                    new_cost = calculate_new_cost(
                        df.loc[i, j],
                        nodes[i_idx],
                        nodes[j_idx],
                        (weights['alpha'], weights['beta'], weights['gamma'], weights['delta'])
                    )
                else: #se a manutenção não foi realizada, não calcula custo
                    continue

                new_df.loc[i, j] = round(new_cost)

    return new_df


# Ler o arquivo CSV diretamente
df = pd.read_csv('matriz_custos_final - matriz_custos_final(1).csv', index_col=0)

# Converter valores para float
for col in df.columns:
    df[col] = pd.to_numeric(df[col].replace('.', '').replace(',', '.'), errors='coerce')

# Processar matriz e gerar novos custos
new_matrix = process_matrix(df)

# Análise comparativa da nova matriz
old_values = df.values.flatten()
old_values = old_values[~np.isnan(old_values)]
new_values = new_matrix.values.flatten()
new_values = new_values[~np.isnan(new_values)]

print("Análise comparativa das matrizes:")
print(f"\nMatriz Original:")
print(f"Mínimo: {min(old_values):,.2f}")
print(f"Máximo: {max(old_values):,.2f}")
print(f"Média: {np.mean(old_values):,.2f}")
print(f"Mediana: {np.median(old_values):,.2f}")

print(f"\nNova Matriz:")
print(f"Mínimo: {min(new_values):,.2f}")
print(f"Máximo: {max(new_values):,.2f}")
print(f"Média: {np.mean(new_values):,.2f}")
print(f"Mediana: {np.median(new_values):,.2f}")

# Exportar nova matriz para CSV
new_matrix.to_csv('nova_matriz_custos.csv')