# Karl-Johan Petersson & Emil Holm

import numpy as np
import matplotlib.pyplot as plt
from travelingsalesman_minheap import TSM_heap
from travelingsalesman import TSM
import time

# Distance table
D = np.array([[0, 12, 26, 6, 16, 10, 10, 18, 12],
              [12, 0, 12, 18, 5, 14, 15, 6, 8],
              [26, 12, 0, 8, 14, 13, 8, 4, 10],
              [6, 18, 8, 0, 18, 12, 8, 3, 5],
              [16, 5, 14, 18, 0, 14, 6, 9, 2],
              [10, 14, 13, 12, 14, 0, 10, 13, 20],
              [10, 15, 8, 8, 6, 10, 0, 9, 10],
              [18, 6, 4, 3, 9, 13, 9, 0, 4],
              [12, 8, 10, 5, 2, 20, 10, 4, 0]], dtype=int)


# Initialize lists to store data points
x_values = []
y_values = []
heap_y_values = []

# Loop through the range of arguments
for S in range(5, 15):
    random_map = np.zeros((S, S), dtype=int)

    for i in range(S):
        for j in range(i+1, S):
            # Generate random distances (range 1 to 100)
            distance = np.random.randint(1, 100)
            # Assign distances symmetrically
            random_map[i, j] = distance
            random_map[j, i] = distance

    D = random_map

    # Normal
    start_time = time.time()
    path, cost = 0, 0 # TSM(D)
    execution_time = time.time() - start_time

    # Heap
    heap_start_time = time.time()
    heap_path = TSM_heap(D)
    heap_execution_time = time.time() - heap_start_time

    x_values.append(S)
    y_values.append(execution_time)
    heap_y_values.append(heap_execution_time)

    print('Cities: ', S)
    print('D\n', D)
    print('NORMAL > time: ', execution_time,  ',\toptimal path: ', path, ', cost: ', cost)
    print('HEAP   > time: ', heap_execution_time,  ',\toptimal path: ', heap_path.x, ', cost: ', heap_path.lb)

# Plot the collected data points
plt.plot(x_values, y_values, marker='o', linestyle='-', label='normal')
plt.plot(x_values, heap_y_values, marker='o', linestyle='--', label='heap')
plt.title('Execution times of traveling salesman algorithms')
plt.xlabel('N cities')
plt.ylabel('time (seconds)')
plt.grid(True)
plt.show()