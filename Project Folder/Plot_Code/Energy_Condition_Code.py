# Physcial Acceptabiliy Line Plots
import numpy as np
from scipy.integrate import odeint, cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = '14'
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.linewidth'] = 1.
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.handletextpad'] = 0.
plt.rcParams["axes.formatter.limits"]= [-2, 4]
plt.ticklabel_format(style='sci', axis='x', scilimits=(0.01,1e4))

# Constants
p0_rho0 = 0.25
rb, L = 1e4, 1e-52

# Ranges for rho0 and Q
rho0_list = [2e-9, 5e-9, 1e-8]
Q_list = [1e3, 3e3, 5e3]

# Integration range
a = 0.01
b = rb
N = 15000
rp = np.linspace(a, b, N)

# Parameters
# =============================================================================
# Gamma_values = np.linspace(1, 5, 50)
# n_vals = np.arange(1, 5.1, 0.1)
# =============================================================================

Gamma_values = np.linspace(1, 5, 100)
n_vals = np.arange(1, 5.1, 0.01)

# Quantities to plot
# =============================================================================
# plot_quantities = [
#     "cs2", "weak", "null", "strong", "dom", "rho", "pres", "mass", "bulk"
# ]
# =============================================================================

plot_quantities = [
    "bulk"
]

# Create dictionaries to hold figures and axes
figs, axes = {}, {}

# Scatter storage
scatter_n = []
scatter_Gamma = []

# Create subplots dynamically
for quantity in plot_quantities:
    if quantity == "rho":
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    else:
        fig, ax = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    figs[quantity] = fig
    axes[quantity] = ax

# ODE function
def f(val, r, n, Gamma, K, Q):
    m, omega = val
    ft = omega
    phinum = 24*np.pi*r**3*K*(omega/(4*np.pi*r**2))**Gamma \
             + 6*m \
             + 6*Q**2*(1-n)*r**(2*n-1)/(rb**(2*n)*(2*n-1)) \
             + 2*L*r**3
    phiden = 6*r**2 - 12*m - 6*Q**2*r**(2*n-1)/(rb**(2*n)*(2*n-1)) + 2*L*r**4
    phip = phinum/phiden
    t1 = 2*omega/r
    t2 = (omega**(2-Gamma)*(4*np.pi*r**2)**(Gamma-1)/(K*Gamma) - omega/Gamma)*phip
    t3 = Q**2*r**(2*n-3)*omega**(1-Gamma)/(K*Gamma*rb**(2*n)*(4*np.pi*r**2)**(1-Gamma))
    fo = t1 - t2 + t3
    return np.array([ft, fo], float)

skip = 15   # use for line plotting only

#  Main loop over parameters
for i, rho0 in enumerate(rho0_list):
    for j, Q in enumerate(Q_list):
        for quantity in plot_quantities:
            if quantity == "rho" and i > 0:
                continue
            if quantity == "rho":
                ax = axes[quantity][j]
            else:
                ax = axes[quantity][i, j]

            rho_str = f"{rho0:.0e}".replace("e", r"\times 10^{") + "}"
            Q_str = f"{Q:.0e}".replace("e", r"\times 10^{") + "}"
            title = f"$\\rho_0 = {rho_str},\\ Q = {Q_str}$"
            ax.set_title(title)
            ax.grid(True)
            ax.tick_params(direction='in')

        line_styles = ['-', '--', ':', '-.']
        style_idx = 0

        for ni, n in enumerate(n_vals):
            for gi, Gamma in enumerate(Gamma_values):
                try:
                    m0 = 1e-12
                    mp0 = 4*np.pi*a**2*rho0
                    yv0 = np.array([m0, mp0], float)
                    K = p0_rho0 / (rho0**(Gamma-1))

                    sol = odeint(f, yv0, rp, args=(n, Gamma, K, Q))
                    mp_sol = sol[:, 0]
                    omega_sol = sol[:, 1]
                    rhor = omega_sol / (4*np.pi*rp**2)

                    pres = K * rhor**Gamma
                    cs2 = K * Gamma * rhor**(Gamma - 1)
                    comp = (rhor*cs2)

                    q_r = Q * (rp / rb)**n
                    e_r = cumulative_trapezoid(q_r**2 / rp**2, rp, initial=0)
                    e_pri = np.gradient(e_r, rp)

                    null = rhor + pres
                    weak = 8*np.pi*rhor + q_r**2 / rp**4 - L
                    dom = 4*np.pi*(rhor - pres) + q_r**2 / rp**4 - L
                    strong = 4*np.pi*(rhor + 3*pres) + q_r**2 / rp**4 + L

                    if np.all(cs2 < 1) and np.any(rhor < rho0) and not np.allclose(cs2, 0.25, rtol=1e-3, atol=1e-5):
                        # Always collect scatter points
                        scatter_n.append(n)
                        scatter_Gamma.append(Gamma)

                        # Only plot lines if both indices respect skip
                        if ni % skip == 0 and gi % skip == 0:
                            linestyle = line_styles[style_idx % len(line_styles)]

                            for quantity in plot_quantities:
                                if quantity == "rho" and i > 0:
                                    continue
                                if quantity == "rho":
                                    ax = axes[quantity][j]
                                else:
                                    ax = axes[quantity][i, j]

                                if quantity == "cs2":
                                    ax.plot(rp, cs2, lw=1, linestyle=linestyle)
                                elif quantity == "weak":
                                    ax.plot(rp, weak, lw=1, linestyle=linestyle)
                                elif quantity == "null":
                                    ax.plot(rp, null, lw=1, linestyle=linestyle)
                                elif quantity == "dom":
                                    ax.plot(rp, dom, lw=1, linestyle=linestyle)
                                elif quantity == "strong":
                                    ax.plot(rp, strong, lw=1, linestyle=linestyle)
                                elif quantity == "rho":
                                    ax.plot(rp, rhor, lw=1, linestyle=linestyle)
                                elif quantity == "pres":
                                    ax.plot(rp, pres, lw=1, linestyle=linestyle)
                                elif quantity == "mass":
                                    ax.plot(rp, mp_sol, lw=1, linestyle=linestyle)
                                elif quantity == "bulk":
                                    ax.plot(rp, comp, lw=1, linestyle=linestyle)

                            style_idx += 1
                except:
                    continue

# Makes scatter plot after loop
fig_scatter, ax_scatter = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
figs["scatter"] = fig_scatter
axes["scatter"] = ax_scatter

for i in range(3):
    for j in range(3):
        ax_scatter[i, j].scatter(scatter_n, scatter_Gamma, color='blue', s=25, marker='o')
        ax_scatter[i, j].grid(True)
        # only put x-labels on the bottom row
        if i == 2 and j==1:
            ax_scatter[i, j].set_xlabel("$n$", fontsize = 18)

        # only put y-labels on the left column
        if i==1 and j == 0:
            ax_scatter[i, j].set_ylabel("$\\Gamma$", fontsize = 18)
        fig_scatter.tight_layout()
        fig_scatter.savefig("scatter_plots.pdf")
        plt.close(fig_scatter) 

# Final formatting
titles = {
    "cs2": "$c_s^2(r)$", "weak": "$T_{\\alpha\\beta}u^\\alpha u^\\beta$", "null": "$T_{\\alpha\\beta}k^\\alpha k^\\beta$",
    "strong": "$(T_{\\alpha\\beta}-\\frac{1}{2}T^\\sigma_\\sigma g_{\\alpha\\beta})$", "dom": "$T_{\\alpha\\beta}v^\\beta T^\\alpha_\\sigma v^\\sigma$",
    "rho": "$\\rho(r)$", "pres": "$p(r)$", "mass": "$m(r)$", "bulk": "$B(r)$", "scatter": ""
}

for q in figs.keys():
    fig = figs[q]
    fig.tight_layout()
    fig.subplots_adjust(left=0.09, top=0.95, bottom=0.06)
    if titles[q] != "":
        fig.text(0.02, 0.5, titles[q], va='center', rotation='vertical', fontsize=18)
        fig.text(0.5, 0.01, "$r$ [m]", ha='center', fontsize=18)
    fig.savefig(f"{q}_plot.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)