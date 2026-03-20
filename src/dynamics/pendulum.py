import numpy as np

def pendulum_cart(t, y, u, params):
    M, m, l, g = params["M"], params["m"], params["l"], params["g"]
    x, theta, dx, dtheta = y

    s = np.sin(theta)
    c = np.cos(theta)

    ddx = (4 * u - 3 * m * g * s * c + 4 * m * l * s * dtheta**2) / (4 * (m + M) - 3 * m * c**2)
    ddtheta = ((M + m) * 3 * g * s - 3 * u * c - 3 * l * m * s * c * dtheta**2) / (4 * l * (m + M) - 3 * l * m * c**2)

    return [dx, dtheta, ddx, ddtheta]