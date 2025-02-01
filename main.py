
import random
import numpy as np 

from generate_points_matrix import generate_matrix
from get_min_distance import get_min_distance
[1, 5, 0, 3, 4, 2, 6]

# Points to visit
POINTS = 7
# Defines influence of Pheromones
ALPHA = 5
# Defines influence of distance between current_node and next_node
BETA = 3
ANT_FACTOR = 1
EVAPORATE_FACTOR = 0.8
RANDOM_CHOICE_FACTOR = 0.05
ITERATIONS = 10
MIN_PHERAMON = 0.001

PHEROMONES = np.ones((POINTS, POINTS))
DISTANCE = generate_matrix(POINTS)


class Ant:
    def __init__(self, points: int):
        self.to_visit = set(range(points))
        self.visited = set()
        self.path = [self._random_node()]
        self.distance = 0 
        # print(self.to_visit, self.visited)
    
    def _get_heuristic(self, next_node: int) -> float:
        distance_to_next_node = DISTANCE[self.path[-1], next_node]
        return 1 / distance_to_next_node 
    
    def _get_probability(self, next_node: int) -> float:
        pheromones = PHEROMONES[self.path[-1], next_node] 
        return (pheromones ** ALPHA) * (self._get_heuristic(next_node) ** BETA)
    
    def _random_node(self) -> int:
        node = random.choice(list(self.to_visit))
        self._add_to_visited(node)
        return node
    
    def _add_to_visited(self, node: int) -> int:
        self.visited.add(node)
        self.to_visit.remove(node)
        return node
        
    def fitness(self) -> float:
        if self.distance > 0:
            return 1 / self.distance
        return 0
    
    def choose_next_node(self):
        
        if len(self.to_visit) == 1:
            node = self.to_visit.pop()
            self.visited.add(node)
            return node
        
        if random.random() < RANDOM_CHOICE_FACTOR:
            return self._random_node()
        
        not_visited_nodes = []
        probabilities = []
        total_probabilities = 0
        
        for next_node in self.to_visit:
            node_probability = self._get_probability(next_node)
            not_visited_nodes.append(next_node)
            probabilities.append(node_probability)
            total_probabilities += node_probability
        
        weights = [p / total_probabilities for p in probabilities]
        
        # print(f"\n{probabilities = }\n{weights = }\n")
        
        node = random.choices(not_visited_nodes, weights=weights, k=1)[0]
        self._add_to_visited(node)

        return node

    def make_move(self):
        next_node = self.choose_next_node()
        self.distance += DISTANCE[self.path[-1], next_node]
        self.path.append(next_node)
        # print(f"\n{self.path = }\n{self.distance = }\n")
        return next_node
        
        
        

def evaporate() -> None:
    global PHEROMONES
    
    mask = PHEROMONES > MIN_PHERAMON
    
    # print(f"{mask = }")
    # print(f"{PHEROMONES[mask] = }")
    # print(f"{PHEROMONES = }")
    
    PHEROMONES[mask] *= EVAPORATE_FACTOR
    


def update_pheromones(population: list[Ant]) -> None:
    global PHEROMONES
    
    for ant in population:
        for i in range(1, len(ant.path)):
            PHEROMONES[ant.path[i - 1], ant.path[i]] += ant.fitness()
            PHEROMONES[ant.path[i], ant.path[i - 1]] += ant.fitness()


def update_best_ant_pheromones(ant: Ant) -> None:
    global PHEROMONES
    
    for i in range(1, len(ant.path)):
        PHEROMONES[ant.path[i - 1], ant.path[i]] += ant.fitness()
        PHEROMONES[ant.path[i], ant.path[i - 1]] += ant.fitness()


def main():
    
    best_ant = None
    
    for iteration in range(ITERATIONS):
        
        population = [Ant(POINTS) for _ in range(POINTS * ANT_FACTOR)]
        
        for _ in range(1, POINTS):
            for ant in population:
                ant.make_move()
            
        temp_best = population[0]
        for ant in population[1:]:
            if ant.distance < temp_best.distance:
                temp_best = ant
        
        if best_ant is None or temp_best.distance < best_ant.distance:
            best_ant = temp_best 
        
        # Evaporate pheromones
        evaporate()
        # Update pheromones from current population
        # update_pheromones(population)
        update_best_ant_pheromones(best_ant)
            
        
        # print(f"\n{iteration = }: {best_ant.distance = }\n")
        print(f"\n{iteration = }: {best_ant.distance = } {best_ant.path = }\n")
    
    print(PHEROMONES)
        
        
    
    
if __name__ == '__main__':
    # Start ACO ----
    main()
    
    # print(f"DISTANCE:\n{DISTANCE}")
    
    res = get_min_distance(DISTANCE)
    print(f"\nPersice {res = }\n")

