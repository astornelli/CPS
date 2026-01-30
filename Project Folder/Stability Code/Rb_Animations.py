import numpy as np
from scipy.integrate import odeint, cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import argrelextrema

# ----------------------------------------------------
# Global constants
# ----------------------------------------------------
p0_rho0 = 0.25
Q = 3e3
rho0 = 5e-9
L = -1e-52
a = 0.01
N = 15000

Gamma_values = np.linspace(1, 5, 100)
n_vals = np.arange(1, 5.1, 0.01)
skip = 3

rb_min, rb_max = 5e3, 1.5e4
rb_vals = np.linspace(rb_min, rb_max, 100)

# ----------------------------------------------------
# ODE system
# ----------------------------------------------------
def f(val, r, n, Gamma, K, rb):
    m, omega = val
    ft = omega
    phinum = (24*np.pi*r**3*K*(omega/(4*np.pi*r**2))**Gamma 
              + 6*m 
              + 6*Q**2*(1-n)*r**(2*n-1)/(rb**(2*n)*(2*n-1)) 
              + 2*L*r**3)
    phiden = (6*r**2 
              - 12*m 
              - 6*Q**2*r**(2*n-1)/(rb**(2*n)*(2*n-1)) 
              + 2*L*r**4)
    phip = phinum/phiden

    t1 = 2*omega/r
    t2 = (omega**(2-Gamma)*(4*np.pi*r**2)**(Gamma-1)/(K*Gamma) - omega/Gamma)*phip
    t3 = Q**2*r**(2*n-3)*omega**(1-Gamma)/(K*Gamma*rb**(2*n)*(4*np.pi*r**2)**(1-Gamma))

    fo = t1 - t2 + t3
    return np.array([ft, fo], float)

# ----------------------------------------------------
# Matplotlib figure setup
# ----------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

scatter_data = {}

ax.set_xlim(0.51, 5.51)
ax.set_ylim(-0.51, 5.5)
ax.grid(True)
ax.set_xlabel("$n$")
ax.set_ylabel("$\\Gamma$")
ax.minorticks_on()
ax.tick_params(which='both', direction='in', top=True, right=True)

# ----------------------------------------------------
# Animation update function
# ----------------------------------------------------
def update(frame):
    rb = rb_vals[frame]

    rp = np.linspace(a, rb, N)

    max_points = []
    min_points = []
    both_points = []

    for n in n_vals[::skip]:
        for Gamma in Gamma_values[::skip]:

            try:
                # Initial conditions
                m0 = 1e-12
                mp0 = 4*np.pi*a**2*rho0
                yv0 = np.array([m0, mp0], float)
                K = p0_rho0 / (rho0**(Gamma - 1))

                # ODE
                sol = odeint(f, yv0, rp, args=(n, Gamma, K, rb))
                m_sol, omega_sol = sol[:,0], sol[:,1]

                rhor = omega_sol / (4*np.pi*rp**2)
                cs2 = K*Gamma*rhor**(Gamma - 1)

                # Electric quantities
                q_r = Q * (rp/rb)**n
                e_r = Q**2 * rp**(2*n-1) / ( rb**(2*n)*(2*n-1) )
                e_pri = np.gradient(e_r, rp)
                

                # Metric
                phinum = (24*np.pi*K*rhor**Gamma*rp**3
                          + 6*m_sol + 3*e_r - 3*rp*e_pri + 2*L*rp**3)
                phidenom = 2*rp*(3*rp - 6*m_sol - 3*e_r + L*rp**3)
                phip = phinum / phidenom
                phi = cumulative_trapezoid(phip, rp, initial=0)

                psidenom = 1 - 2*m_sol/rp - e_r/rp + L*rp**2/3
                psi = 0.5*np.log(1/psidenom)
                eps = 25
                e_eps = 6
                E_r = np.exp(phi + psi) * q_r / (rp**2)
                F_r = cumulative_trapezoid(E_r, rp, initial=0)

                if (not np.all(np.isfinite(phi))) or (not np.all(np.isfinite(psi))):
                    continue

                # Effective potential
                VeffNMs = rp*rp*np.exp(-2*phi)

                if not np.all(np.isfinite(VeffNMs)):
                    continue

                # Maxima detection
                max_idx = argrelextrema(VeffNMs, np.greater, order=10)[0]
                min_idx = argrelextrema(VeffNMs, np.less, order=10)[0]

                # Physically acceptable
                if np.all(cs2 < 1) and np.any(rhor < rho0):
                    has_max = len(max_idx) > 0
                    has_min = len(min_idx) > 0

                    if has_max and has_min:
                        both_points.append((n, Gamma))
                    elif has_max:
                        max_points.append((n, Gamma))
                    elif has_min:
                        min_points.append((n, Gamma))

            except Exception:
                continue

    # Update scatter
    scatter_data[(rho0, Q)] = {
        "max": np.array(max_points),
        "min": np.array(min_points),
        "both": np.array(both_points)
    }
    pts = scatter_data[(rho0, Q)]
    if len(pts["max"]) > 0:
        ax.scatter(pts["max"][:,0], pts["max"][:,1], s=18, color='red', rasterized=True)
        
    if len(pts["min"]) > 0:
        ax.scatter(pts["min"][:,0], pts["min"][:,1], s=18, color='blue', rasterized=True)
    
    if len(pts["both"]) > 0:
        ax.scatter(pts["both"][:,0], pts["both"][:,1], s=25, color='purple', rasterized=True)

    fmrb = f"{rb:.2e}"
    digit, exponent = fmrb.split('e')
    ax.set_title(
        rf"$r_b = {float(digit):.2f} \times 10^{{{int(exponent)}}}$", y = 1.04
    )
    plt.suptitle(r"$(p_0/\rho_0 = 0.25)$", fontsize=12, y=0.91)


# ----------------------------------------------------
# Create and save animation
# ----------------------------------------------------
ani = FuncAnimation(fig, update, frames=len(rb_vals), interval=1000, blit=False)

writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save("rb_animation_NN.mp4", writer=writer)