import numpy as np

def get_min_distance(distance: np.array) -> float:
    
    n = len(distance)
    min_path = (float('inf'), [])
    
    def backtrack(visited: set, path: list[int], path_sum: int = 0):
        nonlocal min_path
        
        if len(path) == n:
            min_path = min(min_path, (path_sum, path[:]), key=lambda x: x[0])
            
        for node in range(n):
            if not node in visited:
                if path:
                    dist_to_node = distance[path[-1], node]
                    path_sum += dist_to_node
                visited.add(node)
                path.append(node)
                backtrack(visited, path, path_sum)
                path.pop()
                visited.remove(node)
                if path:
                    path_sum -= dist_to_node
    
    backtrack(set(), [])
    
    return min_path
                