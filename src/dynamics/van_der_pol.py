def van_der_pol(t, y, mu, u):
    x, dx = y
    ddx = mu * (1 - x**2) * dx - x + u
    return [dx, ddx]