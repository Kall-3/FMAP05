import numpy as np

inf = 10**4
# Example problem
C = np.array([[11, inf, 8,  8],
              [7,    5, 6, 12],
              [10,   6, 8,  5]])
n, m = np.shape(C)

X = np.zeros((n, m))
supply = np.array([100, 120, 60])
demand = np.array([50, 40, 90, 70])

s, d = np.sum(supply), np.sum(demand)

# Another problem
C = np.array([[5, 7,  9, 6],
              [6, 7, 10, 5],
              [7, 6,  8, 1]])
n, m = np.shape(C)

X = np.zeros((n, m))
supply = np.array([120, 140, 100])
demand = np.array([100, 60, 80, 120])

s, d = np.sum(supply), np.sum(demand)

has_dummy = False

#
# Determine supply demand relationship
#

if s < d:
    print("Supply too small, infeasible\n")
elif s > d:
    print("Supply greater than demand, introduce dummy demand")
    new = np.array([0 for _ in range(n)])
    C = np.column_stack((C, new))
    X = np.column_stack((X, new))
    demand = np.append(demand, s - d)
    has_dummy = True
    print("C:\n", C)
    print("demand:\n", demand, '\n')
else:
    print("Supply meets demand\n")


#
# Find initial feasable solution using Vogel's method
#

# Helper function to fill supply/cost table "X"
current_supply_used = [0, 0, 0]

def fill_demand(target, row, current_supply_used): # True for row, false for col
    if row:
        # print("row best")
        C_c = np.copy(C)
        while True:
            min_col = np.argmin(C_c[target, :]) # Best row to fill
            if demand[min_col] - np.sum(X[:, min_col]) != 0:
                break
            else: # Can't choose this, take next best
                C_c[target, min_col] = inf

        # print(target, "row best, min col: ", min_col)
        
        available_supply = supply[target] - current_supply_used[target]
        demanded_supply = demand[min_col] - np.sum(X[:, min_col])

        if available_supply <= demanded_supply:
            X[target, min_col] += available_supply # Fill
            current_supply_used[target] += available_supply
        else: # Surplus after we fill
            X[target, min_col] += demanded_supply
            current_supply_used[target] += demanded_supply
            current_supply_used = fill_demand(target, row, current_supply_used)

    else:
        # Find lowest cost with demand left
        C_c = np.copy(C)
        while True:
            min_row = np.argmin(C_c[:, target]) # Best row to fill
            if current_supply_used[min_row] < supply[min_row]:
                break
            else:
                C_c[min_row, target] = inf
            
        # print(target, "col best, min row: ", min_row)

        available_supply = supply[min_row] - current_supply_used[min_row]
        demanded_supply = demand[target] - sum(X[:, target])

        if available_supply <= demanded_supply:
            X[min_row, target] += available_supply # Fill
            current_supply_used[min_row] += available_supply # Update reduced costs
            current_supply_used = fill_demand(target, row, current_supply_used) # Still have demand
        else: # Surplus after we fill
            X[min_row, target] += demanded_supply
            current_supply_used[min_row] += demanded_supply

    return current_supply_used

# Fill table in order of highest diff, then last row first, then last column first
while np.sum(current_supply_used) < d: # Current supply total less than demand
    # Calculate best difference between second and smallest cost, use largest difference
    best_diff = -1
    best_diff_coords = (-1, -1) # (Best row/col, True if row, false if col, -1/None initially)
    
    for j, col in enumerate(C.T[:-1] if has_dummy else C.T):
        if demand[j] != np.sum(X[:, j]):
            # print(col)
            smallest = None
            second_smallest = None
            for i, cost in enumerate(col):
                if supply[i] != current_supply_used[i]:
                    if smallest == None or cost < smallest:
                        second_smallest = smallest
                        smallest = cost
                    elif second_smallest == None or cost < second_smallest:
                        second_smallest = cost
    
            if smallest and second_smallest != None:
                diff = second_smallest - smallest
                if diff >= best_diff:
                    best_diff = diff
                    best_diff_coords = (j, 1) # col j, axis 1
    
    for i, row in enumerate(C):
        if supply[i] != current_supply_used[i]:
            if has_dummy:
                row = row[:-1]
            # print(row)
            smallest = None
            second_smallest = None
            for j, cost in enumerate(row):
                if demand[j] != np.sum(X[:,j]):
                    if smallest == None or cost < smallest:
                        second_smallest = smallest
                        smallest = cost
                    elif second_smallest == None or cost < second_smallest:
                        second_smallest = cost

            if smallest and second_smallest != None:
                diff = second_smallest - smallest
                if diff >= best_diff:
                    best_diff = diff
                    best_diff_coords = (i, 0) # Row i, axis 0
        
    # print(f'BD: {best_diff}, c: {best_diff_coords}\n')
    current_supply_used = fill_demand(best_diff_coords[0], not best_diff_coords[1], current_supply_used)

# Check for remaining supply and place in dummy demand
surplus_supply = sum(supply) - sum(current_supply_used)
if surplus_supply != 0:
    for i in range(n):
        if current_supply_used[i] != supply[i]:
            X[i,-1] = surplus_supply
        
print(f'X:\n{X}')
print(f'C:\n{C}\n')


# Iterate calculating the reduced costs and move from most expensive to negative reduced costs until no negative remain

while True:
    # Find where the reduced cost is zero, i.e. where the supply is not zero
    non_zero_reduced_cost_coords = []
    for i in range(n):
        for j in range(m): # range(np.shape(X)[1]):
            if X[i, j] > 0:
                non_zero_reduced_cost_coords.append((i,j))
    
    # print(non_zero_reduced_cost_coords)
    
    # Iterate through and calculate all slacks (v & w)
    v = np.zeros(n) # First coord is zero
    w = np.zeros(m)
    
    solved_v = np.copy(v)
    solved_w = np.copy(w)
    
    first = True
    
    while non_zero_reduced_cost_coords:
        coords = non_zero_reduced_cost_coords.pop(0)
        # print(coords)
        # print(f'v: {v}, w: {w}')
        if first:
            # first v = 0
            solved_v[coords[0]] = 1
            first = False
    
        if not solved_v[coords[0]] and solved_w[coords[1]]:
            v[coords[0]] = C[coords] - w[coords[1]]
            solved_v[coords[0]] = 1
        elif solved_v[coords[0]]:
            w[coords[1]] = C[coords] - v[coords[0]]
            solved_w[coords[1]] = 1
        else: # Can't solve right now
            non_zero_reduced_cost_coords.append(coords)
    
    print(v, w)
    
    best_reduced_cost = 0;
    
    # Go through and find if there are any negative relative costs, if so save most negative
    for i in range(n):
        for j in range(m):
            reduced_cost = C[i,j] - v[i] - w[j]
            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
    
    if best_reduced_cost < 0: # Make pivot
        break
    else:
        break # Done!


cost = 0
for i in range(n):
    for j in range(m):
        cost += C[i,j] * X[i,j]

print(cost)
