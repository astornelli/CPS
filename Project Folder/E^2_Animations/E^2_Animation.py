import numpy as np
from scipy.integrate import odeint, cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import argrelextrema

# ----------------------------------------------------
# Global constants
# ----------------------------------------------------
p0_rho0 = 0.25
rb, L = 1e4, -1e-52
rho0 = 5e-9
Q = 3e3
a = 0.01
b = rb
N = 15000
rp = np.linspace(a, b, N)

Gamma_values = np.linspace(1, 5, 100)
n_vals = np.arange(1, 5.1, 0.01)
skip = 3

# ----------------------------------------------------
# ODE system
# ----------------------------------------------------
def f(state, r, n, Gamma, K, Q):
    m, omega = state
    ft = omega
    # numerator and denominator for phip (kept your original form)
    phinum = (24*np.pi*r**3*K*(omega/(4*np.pi*r**2))**Gamma
              + 6*m + 6*Q**2*(1-n)*r**(2*n-1)/(rb**(2*n)*(2*n-1))
              + 2*L*r**3)
    phiden = (6*r**2 - 12*m - 6*Q**2*r**(2*n-1)/(rb**(2*n)*(2*n-1)) + 2*L*r**4)
    # protect against division by zero
    phip = phinum/phiden
    t1 = 2*omega/r
    t2 = (omega**(2-Gamma)*(4*np.pi*r**2)**(Gamma-1)/(K*Gamma) - omega/Gamma)*phip
    t3 = Q**2*r**(2*n-3)*omega**(1-Gamma)/(K*Gamma*rb**(2*n)*(4*np.pi*r**2)**(1-Gamma))
    fo = t1 - t2 + t3
    return np.array([ft, fo], float)

# ----------------------------------------------------
# ε^2 range for animation
# ----------------------------------------------------
eps_min, eps_max = 0.01, 30
eps_vals = np.linspace(eps_min, eps_max, 100)

# ----------------------------------------------------
# Matplotlib figure setup
# ----------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

scatter_s = ax.scatter([], [], marker="o", color='orange', alpha=0.8, rasterized=True)

ax.set_xlim(0.51, 5.51)
ax.set_ylim(-0.51, 5.5)
ax.grid(True)
ax.set_xlabel("$n$")
ax.set_ylabel("$\\Gamma$")
ax.tick_params(direction='in')
ax.tick_params(axis='both', which='minor', direction='in')
ax.tick_params(which='both', top=True, right=True, bottom=True, left=True)
ax.minorticks_on()

# ----------------------------------------------------
# Animation update function
# ----------------------------------------------------
def update(frame):
    eps = eps_vals[frame]      # animated value

    ns, gs = [], []

    for n in n_vals[::skip]:
        for Gamma in Gamma_values[::skip]:
            try:
                # Initial conditions
                m0 = 1e-12
                mp0 = 4*np.pi*a**2*rho0
                yv0 = np.array([m0, mp0], float)
                K = p0_rho0 / (rho0**(Gamma - 1))

                sol = odeint(f, yv0, rp, args=(n, Gamma, K, Q), atol=1e-8, rtol=1e-6)
                m_sol, omega_sol = sol[:, 0], sol[:, 1]

                rhor = omega_sol / (4 * np.pi * rp**2)
                # protect negative densities
                if np.any(rhor <= 0):
                    continue

                cs2 = K * Gamma * rhor**(Gamma - 1)

                # Electric quantities
                q_r = Q* (rp / rb)**n
                e_r = Q**2 / rb**(2*n) * rp**(2*n-1) / (2*n - 1)
                e_pri = np.gradient(e_r, rp)

                # Metric
                phinum = 24*np.pi*K*rhor**Gamma*rp**3 + 6*m_sol + 3*e_r - 3*rp*e_pri + 2*L*rp**3
                phidenom = 2*rp*(3*rp - 6*m_sol - 3*e_r + L*rp**3)
                # protect division by zero
                safe_mask = (phidenom != 0)
                if not np.any(safe_mask):
                    continue
                phip = np.zeros_like(rp)
                phip[safe_mask] = phinum[safe_mask] / phidenom[safe_mask]
                phi = cumulative_trapezoid(phip, rp, initial=0)

                psidenom = 1 - 2*m_sol/rp - e_r/rp + L*rp**2/3
                if np.any(psidenom <= 0):
                    # psi would be complex; skip this parameter set
                    continue
                psi = np.log(1/psidenom) / 2

                if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(psi)):
                    continue
                
                e_eps = 8.3

                # E, F
                E_r = np.exp(phi+psi)*q_r/rp/rp
                F_r = cumulative_trapezoid(E_r, rp, initial=0)

                # Charged Massive Veff
                VeffNMsv = rp*rp*np.exp(-2*phi)-1/eps*rp*rp

                if not np.all(np.isfinite(VeffNMsv)):
                    continue

                # Maxima detection
                max_idx = argrelextrema(VeffNMsv, np.greater, order=10)[0]

                # Physical acceptance
                if np.all(cs2 < 1) and np.any(rhor < rho0) and len(max_idx) > 0:
                    ns.append(n)
                    gs.append(Gamma)

            except Exception:
                continue

    # ----------------------------------------------------
    # Updates scatter
    # ----------------------------------------------------
    if len(ns) > 0:
        pts = np.column_stack((ns, gs))
    else:
        pts = np.empty((0, 2))

    scatter_s.set_offsets(pts)

    # ----------------------------------------------------
    # Title formatting
    # ----------------------------------------------------
    fmeps = f"{eps:.2e}"
    digit, exponent = fmeps.split('e')
    exp_int = int(exponent)
    ax.set_title(f"$\\mathcal{{E}} = {float(digit):.2f} \\times 10^{{{exp_int}}}$", y=1.04)
    plt.suptitle(rf"$(p_0/\rho_0 = 0.25,\; Q = {Q})$", fontsize=12, y=0.91)
    return scatter_s

# ----------------------------------------------------
# Creates animation
# ----------------------------------------------------
ani = FuncAnimation(fig, update, frames=len(eps_vals), interval=1000, blit=False)

writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save("eps_animation_NM.mp4", writer=writer)