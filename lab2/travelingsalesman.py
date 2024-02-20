# Karl-Johan Petersson & Emil Holm

import numpy as np

def TSM(D):
    N, _ = np.shape(D)

    def boundy(x, minmax):
        # sums cost of taken path
        sum_path = np.sum([D[a, b] for (a,b) in zip(x[:-1], x[1:])])

        # pum_path + min bound for future paths AND pum_path + max bound for future paths
        return np.sum(np.array([minmax[:, i] for i in range(len(minmax[0, :])) if i not in x]), axis=0, dtype=int) + np.array([sum_path, sum_path], dtype=int)


    def branchy(x):
        n = len(x)
        m = N - n

        cities_left = np.array([c for c in range(N) if c not in x])

        X = np.zeros((m, n + 1), dtype=int)
        X[:, :-1] = np.array(x)
        X[:, -1] = cities_left

        return X


    def branch_and_bound(x, minmax, fopt):
        bounds = boundy(x, minmax)

        # Recursive stop when we only have 2 nodes left, or/and when bounds are the same
        #bounds[0] == bounds[1] 
        if len(x) == N:
            final_bounds = bounds[0] + D[x[-1], startx]
            x = np.append(x, startx)
            if final_bounds < fopt:
                fopt = final_bounds
        else:
            # Find new paths
            X = branchy(x)
            nn, _ = np.shape(X)
            B = np.zeros((2, len(X)), dtype=int)

            # Find bounds for new paths
            for i in range(nn): # TODO: Correct range?
                B[:, i] = boundy(X[i, :], minmax)

            # Pick optimal extension? Reorder columns
            order = np.argsort(B[0, :], axis=0)
            B = B[:, order]
            X = X[order, :]

            for i in range(nn):
                if B[0, i] < fopt:
                    x_new, fopt_new = branch_and_bound(X[i, :], minmax, fopt)
                    if fopt_new < fopt:
                        fopt = fopt_new
                        x = x_new
                # else prune

        return x, fopt


    # Initialize recursive function
    minmax = np.zeros((2, N), dtype=int)

    for i in range(0, N):
        paths = np.delete(D[i, :], i)
        minmax[0, i] = min(paths)
        minmax[1, i] = max(paths)

    startx = 0 # TODO: 0 or 1? Initial path lenght?
    bounds = boundy([startx], minmax)
    x, fopt = branch_and_bound(np.array([startx]), minmax, bounds[1])

    # Add path back to original city

    return x, fopt