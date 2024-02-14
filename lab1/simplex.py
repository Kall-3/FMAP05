# BY: Karl-Johan Petersson (ka3617pe-s) and Emil Holm (em3756ho-s)

import numpy as np
from fractions import Fraction

def check_basic(A, b, c, basicvars, iterations, z = 0, nbr_non_slack_vars=-1):
    if nbr_non_slack_vars == -1:
        nbr_non_slack_vars = len(c) - len(basicvars)
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
    
    # Initially, non-basic variables are all in the front, and basic in the back
    nbr_basics      = len(basicvars)
    nbr_nonbasics   = len(c) - nbr_basics


    # Make sure basic variables are ordered
    if not np.equal(A[:,basicvars], np.identity(nbr_basics)).all() or not np.equal(c[basicvars], np.zeros(nbr_basics)).all():
        print("Not identity, converting")

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

        # Remove factors on rows above
        for row in reversed(range(1, nbr_basics)):
            for other_row in range(row):
                factor = A[other_row, row + nbr_nonbasics]
                A[other_row,:] -= factor * A[row,:]
                b[other_row] -= factor * b[row]


    # Pivot iterations
    basic_idxs = basicvars
    non_basic_idxs = [i for i in range(len(c)) if i not in basic_idxs]

    for _ in range(iterations):

        # Find pivot column
        max = -1
        max_col_idx = 0
        for col in range(len(A[0])):
            val = A[:,col].reshape(1,nbr_basics) @ c[basic_idxs].reshape(nbr_basics,1)
            diff = c[col] - val

            # if > we will chose first best
            # if >= we will chose last best
            if diff > max:
                max = diff
                max_col_idx = col

        # Find pivot row
        min = float('inf')
        min_row_idx = 0
        for i, (basic, col_val) in enumerate(zip(b, A[:,max_col_idx])):
            if col_val != 0:
                quote = basic / col_val

                # if > chose firs best
                # if >= chose last best
                if quote <= min and quote >= 0:
                    min = basic / col_val
                    min_row_idx = i

        # Normalize the pivot row
        pivot_element = A[min_row_idx, max_col_idx]
        A[min_row_idx, :] /= pivot_element
        b[min_row_idx] /= pivot_element
    
        # Remove normalized pivot row from other rows
        for i in range(len(A)):
            if i != min_row_idx:
                factor = A[i, max_col_idx]
                A[i,:] -= factor * A[min_row_idx,:]
                b[i] -= factor * b[min_row_idx] 

        factor = c[max_col_idx]
        c -= factor * A[min_row_idx,:]
        z -= factor * b[min_row_idx]

        # Swap non-basic and basic variables used in pivot, to get new basics and non-basics
        basic_idxs[min_row_idx] = max_col_idx
        non_basic_idxs = [i for i in range(len(c)) if i not in basic_idxs]

    # Set up tableau
    m, n = A.shape
    tableau = np.zeros((m + 1, n + 1))

    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b.flatten()
    tableau[-1, :-1] = -c.flatten()
    tableau[-1,-1] = -z

    # Solution vector
    x = np.zeros(len(c))

    if (len(c) - nbr_non_slack_vars) != nbr_basics:
        x[basic_idxs] = b
    else:
        for i in basic_idxs:
            if i < nbr_nonbasics:
                row = next((j for j in range(nbr_basics) if A[j, i] == 1), -1)
                x[i] = b[row]

        # Slack variable values
        S = np.subtract(b.T, np.dot(A[:,0:nbr_nonbasics],x[0:nbr_nonbasics].T))

        x[nbr_nonbasics:] = S

    # Check if solution
    solution = all(np.dot(A[0:nbr_basics], x.T) == b)
    # print(f'Sol: {solution}')

    # Check if x is a basic solution
    basic = solution and all(x[non_basic_idxs] == 0)

    # Check if solution is optimal
    optimal = basic and True if np.max(c) <= 0 else False

    # Check if solution is feasible 
    feasible = basic and all(x[basic_idxs] >= 0)

    return x, basic, optimal, feasible, basic_idxs, tableau


# Problem 2
print("\n-------------------- 2. Test case --------------------")
A = np.array([[1, 1, 1, 0, 0],
              [1, 0, 0, 1, 0],
              [8, 20, 0, 0, 1]])
b = np.array([1, 3/4, 10])
c = np.array([2, 1, 0, 0, 0])
basicvars = [2, 3, 4]

x, basic, optimal, feasible, basic_idxs, tableau = check_basic(A, b, c, basicvars, 2)

print("Solution Vector (x):", x, " basic indicies: ", basic_idxs)
print("Basic solution:", basic, "| Optimal Solution:", optimal, "| Feasible Solution:", feasible)
print("Simplex Tableau:")
print(tableau)


# Problem 3
print("\n-------------------- 3. Test case --------------------")
A = np.array([[3, 2, 1, 0, 0],
              [5, 1, 1, 1, 0],
              [2, 5, 1, 0, 1]])
b = np.array([1, 3, 4])
c = np.array([-1, -1, -1, -1, -1])
basicvars = [2, 3, 4]

x, basic, optimal, feasible, basic_idxs, tableau = check_basic(A, b, c, basicvars, 2)

print("Solution Vector (x):", x, " basic indicies: ", basic_idxs)
print("Basic solution:", basic, "| Optimal Solution:", optimal, "| Feasible Solution:", feasible)
print(f'Basic_idxs: {basic_idxs}')
print("Simplex Tableau:")
print(tableau)


# Problem 4, phase I
print("\n-------------------- 4. II Phase problem, phase I --------------------")
A = np.array([[1, 2, 2, 1, 1, 0, 1, 0, 0],
              [1, 2, 1, 1, 2, 1, 0, 1, 0],
              [3, 6, 2, 1, 3, 0, 0, 0, 1]])
b = np.array([12, 18, 24])
c = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
basicvars = [6, 7, 8]

x, basic, optimal, feasible, basic_idxs, tableau = check_basic(A, b, -c, basicvars, 4)

print("Solution Vector (x):", x)
print("Basic solution:", basic, "| Optimal Solution:", optimal, "| Feasible Solution:", feasible)
print("Simplex Tableau:")
print(tableau)

# Problem 4, phase II
print("\n-------------------- 4. II Phase problem, phase II --------------------")
A = np.array([[0.0,  0.0,  1.0, 0.5, 0.0, 0.0],
              [-1.0, -2.0, 0.0, 0.5, 0.0, 1.0],
              [1.0,  2.0,  0.0, 0.0, 1.0, 0.0]])
b = np.array([3.0, 3.0, 6.0])
c = -np.array([1.0, -2.0, -3.0, -1.0, -1.0, 2.0])
z = 0

nbr_non_slack_vars = np.count_nonzero(c)

# Find new objective function
for r, bi in enumerate(basic_idxs):
    factor = (float) (- c[bi] / A[r, bi])
    c += factor * A[r,:]
    z += factor * b[r]

print("New objective function: ", np.append(c, z))

basicvars = basic_idxs

print(f'basic vars in: {basic_idxs}')

x, basic, optimal, feasible, basic_idxs, tableau = check_basic(A, b, -c, basicvars, 1, -z, nbr_non_slack_vars)

print("Solution Vector (x):", x, "basic indicies: ", basic_idxs)
print("Basic solution:", basic, "| Optimal Solution:", optimal, "| Feasible Solution:", feasible)
print("Simplex Tableau:")
print(tableau)


# Problem 5
print("\n-------------------- 5. Try solving unfeasible problem --------------------")
A = np.array([[2.0, -3.0, 2.0, 1.0, 0.0],
              [-1.0, 1.0, 1.0, 0.0, 1.0]])
b = np.array([3.0, 5.0])
c = np.array([3.0, 2.0, 1.0, 0.0, 0.0])
basicvars = [3, 4]

x, basic, optimal, feasible, basic_idxs, tableau = check_basic(A, b, c, basicvars, 0, 0)

print("Solution Vector (x):", x)
print("Basic solution:", basic, "| Optimal Solution:", optimal, "| Feasible Solution:", feasible)
print("Simplex Tableau:")
# print(np.vectorize(lambda x: str(Fraction(x).limit_denominator(10**5)))(tableau))
print (tableau)


# Problem 5
print("\n-------------------- 7. Try solving non basic problem, phase I --------------------")
A = np.array([[-1.0, 1.0, 1.0, 0.0, 0.0],
              [-1.0, 0.0, 0.0, 1.0, 0.0],
              [8.0, -20.0, .0, 0.0, 1.0]])
b = np.array([1.0, .75, 10.0])
c = np.array([1.0, -1.0, .0, .0, .0])
basicvars = [2, 3, 4]

x, basic, optimal, feasible, basic_idxs, tableau = check_basic(A, b, c, basicvars, 2)

print("Solution Vector (x):", x)
print("Basic solution:", basic, "| Optimal Solution:", optimal, "| Feasible Solution:", feasible)
print("Simplex Tableau:")
# print(np.vectorize(lambda x: str(Fraction(x).limit_denominator(10**5)))(tableau))
print (tableau)
