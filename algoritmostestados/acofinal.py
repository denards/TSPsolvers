import numpy as np
import pandas as pd
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import csv
import time



class ACO:
   def __init__(self,
                n_ants: int = 10,
                n_iterations: int = 100,
                decay: float = 0.1,
                alpha: float = 1.0,
                beta: float = 2.0) -> None:
       """
       inicializa os parâmetros do aco


       argumentos:
           n_ants: número de formigas
           n_iterations: número de iterações
           decay: taxa de evaporação do feromônio
           alpha: importância do feromônio
           beta: importância do Custo
       """
       self.n_ants = n_ants
       self.n_iterations = n_iterations
       self.decay = decay
       self.alpha = alpha
       self.beta = beta

   def load_distances(self, filepath: str) -> np.ndarray:
       """
       Lê uma matriz de Custo de um arquivo CSV, ignorando a primeira linha e a primeira coluna.
       Substitui células vazias e valores inválidos por um valor muito grande.
       """
       df = pd.read_csv(filepath)
       # Ignorar primeira linha e coluna
       matrix = df.iloc[1:, 1:].copy()

       # Converter para float, tratando formato brasileiro
       for col in matrix.columns:
           matrix[col] = pd.to_numeric(
               matrix[col].replace('.', '').replace(',', '.'),
               errors='coerce'
           )

       # Substituir NaN por um valor muito grande (mas não infinito)
       big_value = 1e10
       matrix = matrix.fillna(big_value)

       # Converter para array numpy
       return matrix.values.astype(float)

   def initialize_pheromone(self, n_cities: int) -> np.ndarray:
       """inicializa a matriz de feromônios com valores pequenos aleatórios"""
       return np.ones((n_cities, n_cities)) * 0.1

   def calculate_probabilities(self,
                               pheromone: np.ndarray,
                               distances: np.ndarray,
                               current: int,
                               unvisited: list[int]) -> np.ndarray:
       """calcula as probabilidades de mover para cada vértice não visitada"""
       probs = np.zeros(len(unvisited))

       for i, vertice in enumerate(unvisited):
           # Se o Custo for muito grande (era NaN), use probabilidade muito baixa
           if distances[current][vertice] >= 1e10:
               probs[i] = 1e-10
           else:
               # Usa max para evitar divisão por zero
               distance = max(distances[current][vertice], 1e-10)
               probs[i] = (pheromone[current][vertice] ** self.alpha) * \
                          ((1.0 / distance) ** self.beta)

       # Se todas as probabilidades forem muito pequenas
       if np.sum(probs) < 1e-10:
           # Usa distribuição uniforme
           probs = np.ones(len(unvisited)) / len(unvisited)
       else:
           # Normaliza as probabilidades
           probs = probs / np.sum(probs)

       return probs

   def construct_solution(self,
                          pheromone: np.ndarray,
                          distances: np.ndarray) -> tuple[list[int], float]:
       """constrói um tour completo para uma formiga"""
       n_cities = len(distances)
       unvisited = list(range(1, n_cities))
       tour = [0]  # começa da vértice 0
       total_distance = 0.0

       while unvisited:
           current = tour[-1]
           probs = self.calculate_probabilities(pheromone, distances, current, unvisited)
           next_vertice = np.random.choice(unvisited, p=probs)

           tour.append(next_vertice)
           # Se o Custo for muito grande (era NaN), não adiciona ao Custo total
           if distances[current][next_vertice] < 1e10:
               total_distance += distances[current][next_vertice]
           unvisited.remove(next_vertice)

       # Retorna à vértice inicial
       tour.append(0)
       if distances[tour[-2]][0] < 1e10:  # Verifica se a último Custo é válida
           total_distance += distances[tour[-2]][0]

       return tour, total_distance

   def update_pheromone(self,
                        pheromone: np.ndarray,
                        solutions: list[tuple[list[int], float]]) -> np.ndarray:
       """atualiza os níveis de feromônio baseado nas soluções das formigas"""
       # Evaporação
       pheromone *= (1.0 - self.decay)

       # Adiciona novo feromônio
       for tour, distance in solutions:
           if distance > 0:  # Só atualiza se o Custo for válida
               deposit = 1.0 / distance
               for i in range(len(tour) - 1):
                   current = tour[i]
                   next_vertice = tour[i + 1]
                   pheromone[current][next_vertice] += deposit
                   pheromone[next_vertice][current] += deposit

       return pheromone


   def solve(self, filepath: str) -> tuple[list[int], float]:
       """
       resolve o tsp usando otimização por colônia de formigas


       """
       # carrega os Custos
       distances = self.load_distances(filepath)
       #selecionar 17, pegando um subconjunto da matriz de custo
       n_cities = len(distances)

       # inicializa matriz de feromônios
       pheromone = self.initialize_pheromone(n_cities)


       # rastreia a melhor solução
       best_tour = None
       best_distance = float('inf')


       # loop principal
       for iteration in range(self.n_iterations):
           # gera soluções para todas as formigas
           solutions = []
           for ant in range(self.n_ants):
               tour, distance = self.construct_solution(pheromone, distances)
               solutions.append((tour, distance))


               # atualiza a melhor solução
               if distance < best_distance:
                   best_tour = tour
                   best_distance = distance


           # atualiza níveis de feromônio
           pheromone = self.update_pheromone(pheromone, solutions)


           print(f"iteração {iteration + 1}/{self.n_iterations}, menor custo: {best_distance}")


       return best_tour, best_distance


   def plot_solution(self, tour: list[int], distances: np.ndarray) -> None:
       """plota o tour usando uma visualização 2d simples"""
       n_cities = len(distances)


       # cria coordenadas simples para visualização
       coords = np.random.rand(n_cities, 2) * 100


       plt.figure(figsize=(10, 10))


       # plota as vértices
       plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100)


       # plota o tour
       for i in range(len(tour) - 1):
           current = tour[i]
           next_vertice = tour[i + 1]
           plt.plot([coords[current, 0], coords[next_vertice, 0]],
                    [coords[current, 1], coords[next_vertice, 1]], 'b-')


       plt.title('solução do tsp')
       plt.show()


def print_aco_results(best_tour: list, best_distance: float, filepath: str):
    """
    Imprime os resultados do algoritmo ACO de forma organizada.

    Args:
        best_tour: Lista com a sequência de vértices do melhor caminho
        best_distance: Custo total do melhor caminho
        filepath: Caminho do arquivo CSV para obter os rótulos
    """
    import pandas as pd
    from datetime import datetime

    # Lê o CSV para obter os rótulos das vértices
    df = pd.read_csv(filepath)
    vertices_labels = df.iloc[:, 0].tolist()  # Primeira coluna contém os rótulos

    # Formata o cabeçalho
    print("\n" + "=" * 60)
    print(f"{'RESULTADOS DO ALGORITMO ACO':^60}")
    print("=" * 60)

    # Data e hora da execução
    print(f"\nData/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    # Informações do tour
    print(f"\nNúmero de vértices visitadas: {len(best_tour)}")
    print(f"Custo total percorrida: {best_distance:,.2f}")

    # Sequência do tour
    print("\nSequência do caminho:")
    print("-" * 60)
    print(f"{'Ordem':^8} | {'Vértice':^10} | {'Próxima Vértice':^10} | {'Rótulo':^15}")
    print("-" * 60)

    # Imprime cada passo do tour
    for i in range(len(best_tour) - 1):
        current_vertex = best_tour[i]
        next_vertex = best_tour[i + 1]
        current_label = vertices_labels[current_vertex]

        print(f"{i + 1:^8} | {current_vertex:^10} | {next_vertex:^14} | {current_label:^15}")

    # Imprime a última vértice (retorno ao início)
    print(f"{len(best_tour):^8} | {best_tour[-1]:^10} | {best_tour[0]:^14} | {vertices_labels[best_tour[-1]]:^15}")

    print("-" * 60)

    # Resumo do caminho
    print("\nResumo do caminho:")
    path_str = " → ".join(str(vertices_labels[i]) for i in best_tour)
    # Quebra a string em linhas de 70 caracteres para melhor legibilidade
    path_lines = [path_str[i:i + 70] for i in range(0, len(path_str), 70)]
    for line in path_lines:
        print(line)

    print("\n" + "=" * 60 + "\n")



if __name__ == "__main__":
   # cria o ACO
   ACO = ACO(n_ants=20, n_iterations=100, decay=0.8, alpha=1.0, beta=5.0)
# 1 ants, 100 interactions, decay 0.1 alpha=1.0, beta=2.0
# 20 ants, 10 iteractions, decay 0.1 alpha=1.0, beta=2.0
# 332 ants, 10 iteractions, decay 0.1 alpha=1.0, beta=2.0
# 332 ants, 10 iteractions, decay 0.8 alpha=1.0, beta=2.0
# 450 ants, 10 iteractions, decay 0.8 alpha=1.0, beta=2.0
# 450 ants, 10 iteractions, decay 0.8 alpha=1.0, beta=5.0 17 cities

   # resolve o tsp
   filepath = "nova_matriz_custos.csv"
   start_time = time.time()
   best_tour, best_distance = ACO.solve(filepath)
   end_time = time.time()

   execution_time = end_time - start_time
   print(f'Tempo de Execução: {execution_time:.2f}')



   print("\nMelhor tour encontrado:", best_tour)
   print("Custo Total (algoritmo aco):", best_distance)
   print_aco_results(best_tour, best_distance, filepath)

   # plota a solução
   distances = ACO.load_distances(filepath)
   # ACO.plot_solution(best_tour, distances)

