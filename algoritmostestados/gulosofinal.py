import csv
import networkx as nx
import numpy as np
import time

def read_distance_matrix(csv_file):
    """
    Lê uma matriz de distância de um arquivo CSV, ignorando a primeira linha e a primeira coluna.
    Retorna uma matriz numpy com np.nan para células vazias.
    """
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        # Pular primeira linha
        next(reader)
        # Processar as demais linhas
        matrix = []
        for row in reader:
            # Ignorar primeira coluna e converter valores
            row_values = []
            for val in row[1:]:
                try:
                    if val.strip():  # Se não está vazio
                        # Converter string para float, tratando formato brasileiro
                        val = float(val.replace('.', '').replace(',', '.'))
                    else:
                        val = np.nan
                except (ValueError, AttributeError):
                    val = np.nan
                row_values.append(val)
            matrix.append(row_values)

    return np.array(matrix)

def greedy_tsp(distance_matrix):
    """
    Resolve o problema do caixeiro viajante usando o Algoritmo Guloso.
    """
    num_nodes = distance_matrix.shape[0]
    G = nx.complete_graph(num_nodes)

    # Atribuir pesos às arestas com base na matriz de distância
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                G[i][j]['weight'] = distance_matrix[i, j]

    # Algoritmo Guloso para o TSP
    visited = set()
    route = []
    current_node = 0  # Começa no primeiro nó
    visited.add(current_node)
    route.append(current_node)

    while len(visited) < num_nodes:
        # Encontrar o vizinho mais próximo não visitado
        neighbors = [(neighbor, G[current_node][neighbor]['weight']) for neighbor in G[current_node] if neighbor not in visited]
        next_node = min(neighbors, key=lambda x: x[1])[0]
        route.append(next_node)
        visited.add(next_node)
        current_node = next_node

    # Retornar ao ponto de partida
    route.append(route[0])
    return route


def print_greedy_results(best_tour: list, total_distance: float, filepath: str):
    """
    Imprime os resultados do algoritmo Guloso de forma organizada.

    Args:
        best_tour: Lista com a sequência de vértices do melhor caminho
        total_distance: Distância total do melhor caminho
        filepath: Caminho do arquivo CSV para obter os rótulos
    """
    import pandas as pd
    from datetime import datetime

    # Lê o CSV para obter os rótulos das vértices
    df = pd.read_csv(filepath)
    vertices_labels = df.iloc[:, 0].tolist()  # Primeira coluna contém os rótulos

    # Formata o cabeçalho
    print("\n" + "=" * 70)
    print(f"{'RESULTADOS DO ALGORITMO GULOSO (NEAREST NEIGHBOR)':^70}")
    print("=" * 70)

    # Data e hora da execução
    # print(f"\nData/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    # Informações gerais
    print(f"\nInformações do Tour:")
    print("-" * 70)
    print(f"• Número de vértices visitadas: {len(best_tour)}")
    print(f"• Custo total do tour: {total_distance:,.2f}")

    # Detalhes do caminho
    print("\nDetalhamento do Percurso:")
    print("-" * 70)
    print(f"{'Passo':^8} | {'De':^15} | {'Para':^15} | {'Rótulo De':^12} | {'Rótulo Para':^12}")
    print("-" * 70)

    # Imprime cada passo do tour
    for i in range(len(best_tour) - 1):
        current = best_tour[i]
        next_vertex = best_tour[i + 1]
        current_label = vertices_labels[current]
        next_label = vertices_labels[next_vertex]

        print(f"{i + 1:^8} | {current:^15} | {next_vertex:^15} | {current_label:^12} | {next_label:^12}")

    print("-" * 70)

    # Visualização do caminho completo
    print("\nCaminho Completo:")
    print("-" * 70)

    # Quebra o caminho em linhas para melhor visualização
    path = " → ".join(str(vertices_labels[v]) for v in best_tour)
    max_line_length = 60

    # Quebra o caminho em múltiplas linhas se necessário
    words = path.split(" → ")
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 4 > max_line_length:  # 4 é o comprimento de " → "
            print(" → ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 4

    if current_line:
        print(" → ".join(current_line))

    # Estatísticas adicionais
    print("\nEstatísticas Adicionais:")
    print("-" * 70)
    print(f"• Vértice inicial: {vertices_labels[best_tour[0]]}")
    print(f"• Vértice final: {vertices_labels[best_tour[-1]]}")
    print(f"• Média de custo por trecho: {total_distance / (len(best_tour) - 1):,.2f}")

    print("\n" + "=" * 70 + "\n")


# Exemplo de uso:
if __name__ == "__main__":
    # Assumindo que você já tem os resultados do algoritmo guloso
    filepath = "nova_matriz_custos.csv"

def main():

    csv_file = "nova_matriz_custos.csv"  # Nome do arquivo CSV
    distance_matrix = read_distance_matrix(csv_file)
    # Resolver o TSP
    start_time = time.time()
    route = greedy_tsp(distance_matrix)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f'Tempo de Execução: {execution_time}')

    # print("Melhor tour encontrado (Algoritmo Guloso):", route)
    # print("Distância total: (Algoritmo Guloso):", sum(route))
    print_greedy_results(route, sum(route), filepath)

if __name__ == "__main__":
    main()
