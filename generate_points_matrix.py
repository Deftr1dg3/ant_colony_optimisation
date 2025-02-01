
import numpy as np


def generate_matrix(points: int, max_distance: int | float = 10_000) -> list[list[int]]:
    if points == 0:
        raise ValueError("Unable to create 2d array for zero points.")
    
    matrix = np.triu(np.random.randint(1, max_distance, (points, points)), k=1)
    matrix += matrix.T
    
    return matrix 

