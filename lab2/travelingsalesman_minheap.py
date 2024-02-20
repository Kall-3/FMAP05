# Karl-Johan Petersson & Emil Holm

import numpy as np
import heapq
from functools import total_ordering

@total_ordering
class Path:
    
    def __init__(self, x, lb, ub):
        self.x = x
        self.lb = lb
        self.ub = ub
    
    def __eq__(self, other):
        return self.lb == other.lb
    
    def __lt__(self, other):
        return self.lb < other.lb

def TSM_heap(D):
    # Keeping track of a path

    N, _ = np.shape(D)

    def boundy(x, minmax):
        # sums cost of taken path
        sum_path = np.sum([D[a, b] for (a,b) in zip(x[:-1], x[1:])])

        if len(x) == N:
            return np.array([sum_path + D[x[-1], startx], sum_path + D[x[-1], startx]], dtype=int)
        else:
            # sum_path + min bound for future paths AND sum_path + max bound for future paths
            return np.sum(np.array([minmax[:, i] for i in range(len(minmax[0, :])) if i not in x or i == startx]), axis=0) + np.array([sum_path, sum_path], dtype=int)


    def branchy(x):
        n = len(x)
        m = N - n

        cities_left = np.array([c for c in range(N) if c not in x])

        X = np.zeros((m, n + 1), dtype=int)
        X[:, :-1] = np.array(x)
        X[:, -1] = cities_left

        return X


    # Calculate static minmax
    minmax = np.zeros((2, N), dtype=int)

    for i in range(0, N):
        paths = np.delete(D[i, :], i)
        minmax[0, i] = min(paths)
        minmax[1, i] = max(paths)

    # Calculate original node
    startx = 0
    bounds = boundy(np.array([startx]), minmax)
    lowest_upper_bound = bounds[1]

    root_path = Path([startx], bounds[0], bounds[1])

    best_found_path = Path([], float('inf'), 0)
    potential_paths = [root_path]
    heapq.heapify(potential_paths)

    while len(potential_paths) != 0:
        current_path = heapq.heappop(potential_paths)

        if current_path.lb > lowest_upper_bound:
            # prune
            continue

        if len(current_path.x) == N:
            if current_path.lb < best_found_path.lb:
                best_found_path = current_path
                best_found_path.x = np.append(best_found_path.x, startx)
                continue
            
        # Find new paths
        X = branchy(current_path.x) 
        nn, _ = np.shape(X)

        for i in range(nn):
            bounds = boundy(X[i, :], minmax)

            if bounds[1] < lowest_upper_bound:
                lowest_upper_bound = bounds[1]

            # Only add the new path if its best solution (lb) is better than the most promising paths worst solution (up)

            heapq.heappush(potential_paths, Path(X[i, :], bounds[0], bounds[1]))

    return best_found_path # TODO: expected to return values?

