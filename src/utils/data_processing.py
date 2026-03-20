import numpy as np

# Hankel matrice for Pendulum on a cart example
def Hankel_pendulum(U_data, X_data, N):
    for k in range(0, len(X_data[:, 0, 0])):
        print(f"k = {k} / {len(X_data[:,0,0])-1}")
        for j in range(0, len(X_data[0, :, 0]) - N):
            for i in range(0, N + 1):
                if i == 0:
                    Xi = X_data[k, i + j, :]

                    Xi_ext = [
                        X_data[k, i + j, 0],
                        np.cos(X_data[k, i + j, 1]),
                        np.sin(X_data[k, i + j, 1]),
                        X_data[k, i + j, 2],
                        X_data[k, i + j, 3],
                    ]

                else:
                    Xi_ext_i = [
                        X_data[k, i + j, 0],
                        np.cos(X_data[k, i + j, 1]),
                        np.sin(X_data[k, i + j, 1]),
                        X_data[k, i + j, 2],
                        X_data[k, i + j, 3],
                    ]
                    Xi_ext = np.concatenate((Xi_ext, Xi_ext_i))
                    Xi = np.concatenate((Xi, X_data[k, i + j, :]))
            if j == 0:
                Xj = Xi_ext
            else:
                Xj = np.column_stack((Xj, Xi_ext))
        if k == 0:
            Xk = Xj
        else:
            Xk = np.column_stack((Xk, Xj))

    X0 = Xk[0:5, :] # type: ignore
    Y = Xk[5:, :] # type: ignore

    for k in range(0, len(U_data[:, 0])):
        print(f"k = {k} / {len(U_data[:,0])-1}")
        for j in range(0, len(U_data[0, :]) - N + 1):
            for i in range(0, N):
                if i == 0:
                    Ui = U_data[k, i + j]
                else:
                    Ui = np.append(Ui, U_data[k, i + j])
            if j == 0:
                Uj = Ui
            else:
                Uj = np.column_stack((Uj, Ui))
        if k == 0:
            U = Uj
        else:
            U = np.column_stack((U, Uj))
    return U, X0, Y