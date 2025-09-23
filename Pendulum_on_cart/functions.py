import numpy as np  # Add this import statement
import casadi as ca

# Crest factor calculation
crest_factor = lambda uk: np.max(np.abs(uk)) / np.sqrt(np.mean(uk**2))

# Duplicate function
duplicate = lambda uk, n: np.concatenate([uk] * n)


# Multisine signal generator
def multisine(
    N_points_per_period,
    N_periods=1,
    pmin=1,
    pmax=21,
    prule=lambda p: p % 2 == 1 and p % 6 != 1,
    par=None,
    n_crest_factor_optim=1,
    seed=None,
):
    """A multi-sine generator with only odd frequencies and random phases."""

    if isinstance(seed, int) or seed is None:
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    assert isinstance(rng, np.random.mtrand.RandomState)

    assert pmax < N_points_per_period // 2

    # Crest factor optimization
    if n_crest_factor_optim > 1:
        ybest = None
        crest_best = float("inf")
        for i in range(n_crest_factor_optim):
            seedi = None if seed is None else seed + i
            uk = multisine(
                N_points_per_period,
                N_periods=1,
                pmax=pmax,
                pmin=pmin,
                prule=prule,
                n_crest_factor_optim=1,
                seed=seedi,
            )
            crest = crest_factor(uk)
            if crest < crest_best:
                ybest = uk
                crest_best = crest
        return duplicate(ybest, N_periods)

    N = N_points_per_period
    uf = np.zeros((N,), dtype=complex)
    for p in range(pmin, pmax) if par is None else par:
        if par is None and not prule(p):
            continue
        uf[p] = np.exp(1j * rng.uniform(0, np.pi * 2))
        uf[N - p] = np.conjugate(uf[p])

    uk = np.real(np.fft.ifft(uf / 2) * N)
    uk /= np.std(uk)

    return duplicate(uk, N_periods)


# Define the Van der Pol oscillator
def van_der_pol(t, y, mu, u):
    x, dx = y
    ddx = mu * (1 - x**2) * dx - x + u
    return [dx, ddx]


# Define the Four tank
def four_tank(t, x, u1, u2, params):
    x1, x2, x3, x4 = x
    a1, a2, a3, a4 = params["a1"], params["a2"], params["a3"], params["a4"]
    Sc, gamma_a, gamma_b = params["Sc"], params["gamma_a"], params["gamma_b"]
    g = params["g"]
    dx1 = (
        (-a1 / Sc) * np.sqrt(max(0, 2 * g * x1))
        + (a3 / Sc) * np.sqrt(max(0, 2 * g * x3))
        + gamma_a / (3600 * Sc) * u1
    )
    dx2 = (
        (-a2 / Sc) * np.sqrt(max(0, 2 * g * x2))
        + (a4 / Sc) * np.sqrt(max(0, 2 * g * x4))
        + gamma_b / (3600 * Sc) * u2
    )
    dx3 = (-a3 / Sc) * np.sqrt(max(0, 2 * g * x3)) + (1 - gamma_b) / (3600 * Sc) * u2
    dx4 = (-a4 / Sc) * np.sqrt(max(0, 2 * g * x4)) + (1 - gamma_a) / (3600 * Sc) * u1
    return [dx1, dx2, dx3, dx4]


# define pendulum on a cart
def penulum_cart(t, y, u, params):
    M, m, l, g = params["M"], params["m"], params["l"], params["g"]

    x, theta, dx, dtheta = y

    s = np.sin(theta)
    c = np.cos(theta)

    ddx = (4 * u - 3 * m * g * s * c + 4 * m * l * s * dtheta**2) / (
        4 * (m + M) - 3 * m * c**2
    )
    ddtheta = ((M + m) * 3 * g * s - 3 * u * c - 3 * l * m * s * c * dtheta**2) / (
        4 * l * (m + M) - 3 * l * m * c**2
    )

    return [dx, dtheta, ddx, ddtheta]


# Define feedforward Neural Netowrk
def NN(u, Network, sigma):
    """Evaluate the neural network with input u."""
    y = u
    L = len(Network.linears)
    for i in range(L - 1):
        W = ca.DM(Network.linears[i].weight.detach().cpu().numpy())
        b = ca.DM(Network.linears[i].bias.detach().cpu().numpy())
        y = sigma(W @ y + b)  # @ is matrix multiplication

    W = ca.DM(Network.linears[-1].weight.detach().cpu().numpy())
    b = ca.DM(Network.linears[-1].bias.detach().cpu().numpy())
    y = W @ y + b
    return y


# define sigmoid function using casadi
def sigmoid_ca(x):
    return 1 / (1 + ca.exp(-x))


# LSTM cell functions
def lstm_cell_ca(x_t, h_prev, c_prev, W_ih, W_hh, b, W_fc, b_fc):
    hidden_size = h_prev.shape[0]

    # Compute gates using CasADi matrix operations
    gates = ca.mtimes(W_ih, x_t) + ca.mtimes(W_hh, h_prev) + b

    # Split gates
    i = sigmoid_ca(gates[0:hidden_size])
    f = sigmoid_ca(gates[hidden_size : 2 * hidden_size])
    g = ca.tanh(gates[2 * hidden_size : 3 * hidden_size])
    o = sigmoid_ca(gates[3 * hidden_size : 4 * hidden_size])

    c_t = f * c_prev + i * g
    h_t = o * ca.tanh(c_t)

    # Output layer
    y_t = ca.mtimes(W_fc, h_t) + b_fc

    return y_t, h_t, c_t


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

    X0 = Xk[0:5, :]
    Y = Xk[5:, :]

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
