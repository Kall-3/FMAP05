import numpy as np

def check_basic(A, b, c, basicvars, iterations):
    # INPUT: A - mxn matrix
    #        b - mx1 matrix
    #        c - nx1 matrix
    #        basicvars - list of m indices between 1 and n.
    # OUTPUT
    #        x - solution vector
    #        basic - basic variables
    #        optimal - 1 if x is an optimal solution
    #        feasible - 1 if x is a feasible solution

    # Cast to float for precision and initialize max objective function result z
    A = A.astype(float)
    b = b.astype(float)
    c = c.astype(float)
    z = 0
    
    # Initially, non-basic variables are all in the front, and basic in the back
    nbr_basics      = len(basicvars)
    nbr_nonbasics   = len(c) - nbr_basics

    # Make sure basic variables are ordered
    if not np.equal(A[:,basicvars], np.identity(len(basicvars))).all():
        print(f'\nNot identity matrix\n{A[:,basicvars]}\n')

        # Gaussian elimination to get identity matrix
        for row in range(nbr_basics):
            # Normalize diagonal
            a = A[row, row + nbr_nonbasics]
            A[row,:] /= a

            # Remove factor from rows below
            for other_row in range(row + 1, nbr_basics):
                factor = A[other_row, row + nbr_nonbasics]
                A[other_row,:] -= factor * A[row,:]
                b[other_row] -= factor * b[row]

            factor = c[row + nbr_nonbasics]
            c -= factor * A[row,:]
            z += b[row]

        for row in reversed(range(1, nbr_basics)):
            for other_row in range(row):
                factor = A[other_row, row + nbr_nonbasics]
                A[other_row,:] -= factor * A[row,:]
                b[other_row] -= factor * b[row]
            


    basic_idxs = basicvars
    non_basic_idxs = [x for x in range(len(c)) if x not in basic_idxs]

    for _ in range(iterations):
        # Find pivot column

        # Find optimal pivot column by finding steepest edge
        max = -1
        max_col_idx = 0
        for col in range(len(A[0])):
            val = A[:,col].reshape(1,nbr_basics) @ c[basic_idxs].reshape(nbr_basics,1)
            diff = c[col] - val
            if diff > max:
                max = diff
                max_col_idx = col

        # Find pivot row
        min = float('inf')
        min_row_idx = 0
    
        for i, (basic, col_val) in enumerate(zip(b, A[:,max_col_idx])):
            if col_val != 0 and basic / col_val < min:
                min = basic / col_val
                min_row_idx = i
        
        # Normalize the pivot row
        pivot_element = A[min_row_idx, max_col_idx]
        A[min_row_idx, :] = A[min_row_idx, :] / pivot_element
        b[min_row_idx] /= pivot_element
    
        # Remove normalized pivot row from other rows
        for i in range(len(A)):
            if i != min_row_idx:
                factor = A[i, max_col_idx]
                A[i, :] -= factor * A[min_row_idx, :]
                b[i] -= factor * b[min_row_idx] 

        factor = c[max_col_idx]
        c -= factor * A[min_row_idx,:]
        z -= factor * b[min_row_idx]

        # Swap non-basic and basic variables used in pivot
        basic_idxs.remove(min_row_idx + nbr_nonbasics)
        basic_idxs.append(max_col_idx)


    m, n = A.shape
    tableau = np.zeros((m + 1, n + 1))

    # Set up tableau
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b.flatten()
    tableau[-1, :-1] = -c.flatten()
    tableau[-1,-1] = -z

    
   # Extract basic solution
    x = np.zeros(n)
    x[basicvars] = b

    # Check if x is a basic solution
    basic = True
    for i, c in enumerate(np.array(A).T):
        if i not in basicvars:
            if sum(c) != 1 or not len([num for num in c if num == 0]) == len(c) - 1:
                basic = False
    

    # Check for optimality and feasibility
    optimal = False if np.max(tableau[-1, :-1]) >= 0 else True
    feasible = True if np.all(b >= 0) else False

    return x, basic, optimal, feasible, tableau


# Problem 2

A_start = np.array([[1, 1, 1, 0, 0],
                    [1, 0, 0, 1, 0],
                    [8, 20, 0, 0, 1]])
b_start = np.array([1, 3/4, 10])
c_start = np.array([2, 1, 0, 0, 0])

basicvars_arr = [2, 3, 4]

x, basic, optimal, feasible, tableau = check_basic(A_start, b_start, c_start, basicvars_arr, 2)

print()
# print("Solution Vector (x):", x)
# print("Basic solution:", basic)
# print("Optimal Solution:", optimal)
# print("Feasible Solution:", feasible)
print("Simplex Tableau:")
print(tableau)


# Problem 3

A = np.array([[3, 2, 1, 0, 0],
              [5, 1, 1, 1, 0],
              [2, 5, 1, 0, 1]])
b = np.array([1, 3, 4])
c = np.array([-1, -1, -1, -1, -1])

basicvars = [2, 3, 4]

x, basic, optimal, feasible, tableau = check_basic(A, b, c, basicvars, 1)

print("Simplex Tableau:")
print(tableau)
