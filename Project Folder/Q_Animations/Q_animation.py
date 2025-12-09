import numpy as np
from scipy.integrate import odeint, cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import argrelextrema

# ----------------------------------------------------
# Global constants
# ----------------------------------------------------
p0_rho0 = 0.25
rb = 1e4
rho0 = 5e-9
L = -1e-52
a = 0.01
b = rb
N = 15000
rp = np.linspace(a, b, N)

Gamma_values = np.linspace(1, 5, 100)
n_vals = np.arange(1, 5.1, 0.01)
skip = 3
Q_min, Q_max = 1e1, 1e4
Q_vals = np.linspace(Q_min, Q_max, 100)
# ----------------------------------------------------
# ODE system
# ----------------------------------------------------
def f(val, r, n, Gamma, K, Q):
    m, omega = val
    ft = omega
    phinum = 24*np.pi*r**3*K*(omega/(4*np.pi*r**2))**Gamma \
             +6*m +6*Q**2*(1-n)*r**(2*n-1)/(rb**(2*n)*(2*n-1)) +2*L*r**3
    phiden = 6*r**2-12*m-6*Q**2*r**(2*n-1)/(rb**(2*n)*(2*n-1))+2*L*r**4
    phip = phinum/phiden
    t1 = 2*omega/r
    t2 = (omega**(2-Gamma)*(4*np.pi*r**2)**(Gamma-1)/(K*Gamma)-omega/Gamma)*phip
    t3 = Q**2*r**(2*n-3)*omega**(1-Gamma)/(K*Gamma*rb**(2*n)*(4*np.pi*r**2)**(1-Gamma))
    fo = t1 - t2 + t3
    return np.array([ft, fo], float)

# ----------------------------------------------------
# Matplotlib figure setup
# ----------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

scatter_s = ax.scatter([], [], marker="o", color='blue', alpha=0.8, rasterized=True)

ax.set_xlim(0.51, 5.51)
ax.set_ylim(-0.51, 5.5)
ax.grid(True)
ax.set_xlabel("$n$")
ax.set_ylabel("$\\Gamma$")
ax.tick_params(direction='in')  
ax.tick_params(direction='in')
ax.tick_params(axis='both', which='minor', direction='in')
ax.tick_params(
    which='both',
    top=True,   
    right=True,  
    bottom=True, 
    left=True,
)
ax.minorticks_on()

# ----------------------------------------------------
# Animation update function
# ----------------------------------------------------
def update(frame):
    Q = Q_vals[frame]  

    ns, gs = [], []

    for n in n_vals[::skip]:
        for Gamma in Gamma_values[::skip]:
            try:
                # Initial conditions
                m0 = 1e-12
                mp0 = 4*np.pi*a**2*rho0
                yv0 = np.array([m0, mp0], float)
                K = p0_rho0 / (rho0**(Gamma - 1))

                sol = odeint(f, yv0, rp, args=(n, Gamma, K, Q))
                mp_sol, omega_sol = sol[:, 0], sol[:, 1]

                rhor = omega_sol / (4 * np.pi * rp**2)
                cs2 = K * Gamma * rhor**(Gamma - 1)

                # Electric quantities
                q_r = Q * (rp / rb)**n
                e_r = Q**2 / rb**(2*n) * rp**(2*n-1) / (2*n - 1)
                e_pri = np.gradient(e_r, rp)

                # Metric
                phinum = 24*np.pi*K*rhor**Gamma*rp**3 + 6*mp_sol + 3*e_r - 3*rp*e_pri + 2*L*rp**3
                phidenom = 2*rp*(3*rp - 6*mp_sol - 3*e_r + L*rp**3)
                phip = phinum / phidenom
                phi = cumulative_trapezoid(phip, rp, initial=0)

                psidenom = 1 - 2*mp_sol/rp - e_r/rp + L*rp**2/3
                psi = np.log(1/psidenom) / 2

                if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(psi)):
                    continue

                # E, F
                eps = 5
                e_eps = 8.3
                E_r = np.exp(phi + psi) * q_r / (rp**2)
                F_r = cumulative_trapezoid(E_r, rp, initial=0)

                # Veff
                VeffNMs = rp*rp*np.exp(-2*phi)
                
                if not np.all(np.isfinite(VeffNMs)):
                    continue

                # Maxima detection
                max_idx = argrelextrema(VeffNMs, np.greater, order=10)[0]

                # Physical acceptance
                if np.all(cs2 < 1) and np.any(rhor < rho0) and len(max_idx) > 0:
                    ns.append(n)
                    gs.append(Gamma)

            except Exception:
                continue

    # ----------------------------------------------------
    # Update scatter safely
    # ----------------------------------------------------
    if len(ns) > 0:
        pts = np.column_stack((ns, gs))
    else:
        pts = np.empty((0, 2))

    scatter_s.set_offsets(pts)

    # ----------------------------------------------------
    # Title formatting
    # ----------------------------------------------------
    fmL = f"{Q:.2e}"
    digit, exponent = fmL.split('e')
    ax.set_title(
        rf"$Q = {float(digit):.2f} \times 10^{{{int(exponent)}}}$", y = 1.04
    )
    plt.suptitle(r"$(p_0/\rho_0 = 0.25)$", fontsize=12, y=0.91)

    return scatter_s

# ----------------------------------------------------
# Create animation
# ----------------------------------------------------
ani = FuncAnimation(fig, update, frames=len(Q_vals), interval=1000, blit=False)

writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save("Q_animation_NN.mp4", writer=writer)