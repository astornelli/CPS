# Veff Line Plot
import numpy as np
from scipy.integrate import odeint, cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import argrelextrema

plt.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = '12'
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.linewidth'] = 1.
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.handletextpad'] = 0.3
plt.rcParams["axes.formatter.limits"]= [-2, 4]
plt.ticklabel_format(style='sci', axis='x', scilimits=(0.01,1e4))
plt.ticklabel_format(style='sci', axis='y')

# Constants
p0_rho0 = 0.25
rb, L = 1e4, 1e-52

# Ranges for rho0 and Q
rho0_list = [2e-9, 5e-9, 1e-8]
Q_list = [1e3, 3e3, 5e3]

# Integration range
a = 0.1
b = rb
N = 15000
rp = np.linspace(a, b, N)

# Parameters
Gamma_values = np.linspace(1, 5, 50)
n_vals = np.arange(1, 5.1, 0.1)

# Set up all subplot grids (3x3) for each quantity
# =============================================================================
# plot_quantities = [
#     "VeNN", "VeNM"
# ]
# =============================================================================

plot_quantities = [
    "VeCN", "VeCM"
]

# =============================================================================
# plot_quantities = [
#     "EPhi", "EPsi", "SC"
# ]
# =============================================================================

# Create dictionaries to hold figures and axes
figs, axes = {}, {}

for quantity in plot_quantities:
    fig, ax = plt.subplots(3, 3, figsize=(16, 16), sharex=True)
    figs[quantity] = fig
    axes[quantity] = ax
    
# ODE function
def f(val, r, n, Gamma, K, Q):
    m, omega = val
    ft = omega
    phinum = 24*np.pi*r**3*K*(omega/(4*np.pi*r**2))**Gamma\
             +6*m\
             +6*Q**2*(1-n)*r**(2*n-1)/(rb**(2*n)*(2*n-1))\
             +2*L*r**3
    phiden = 6*r**2-12*m-6*Q**2*r**(2*n-1)/(rb**(2*n)*(2*n-1))+2*L*r**4
    phip = phinum/phiden
    t1 = 2*omega/r
    t2 = (omega**(2-Gamma)*(4*np.pi*r**2)**(Gamma-1)/(K*Gamma)-omega/Gamma)*phip
    t3 = Q**2*r**(2*n-3)*omega**(1-Gamma)/(K*Gamma*rb**(2*n)*(4*np.pi*r**2)**(1-Gamma))
    fo = t1-t2+t3
    return np.array([ft, fo], float)

scatter_n = []
scatter_Gamma = []
skip = 18

# Outer loop: iterate over rho0 and Q
for i, rho0 in enumerate(rho0_list):
    for j, Q in enumerate(Q_list):
        
        # Select correct subplot axes
        axs = {q: axes[q][i, j] for q in plot_quantities}
        
        rho_label = f"{rho0:.0e}"
        mantissa, exp = rho_label.split("e")
        exp = int(exp)
        rho_str = rf"{mantissa}\times 10^{{{exp}}}"
        
        Q_label = f"{Q:.0e}"
        mantissa1, exp1 = Q_label.split("e")
        exp1 = int(exp1)
        Q_str = rf"{mantissa1}\times 10^{{{exp1}}}"
        
        titlerho = f"$\\rho_0 = {rho_str}$"
        titleQ= f"$Q = {Q_str}$"
        
        for q in plot_quantities:
            if j == 0:
                axs[q].set_ylabel(titlerho, fontsize = 16)
            if i == 0:
                axs[q].set_title(titleQ, rotation = 'horizontal', fontsize = 16)
            axs[q].grid(True)
            axs[q].tick_params(direction='in')  
            axs[q].grid(True)
            axs[q].tick_params(direction='in')
            axs[q].tick_params(axis='both', which='minor', direction='in')
            axs[q].tick_params(
                which='both', 
                top=True,     
                right=True,   
                bottom=True,   
                left=True,   
            )
            axs[q].minorticks_on()
            axs[q].set_xlim(0, 1e4)
            axs[q].ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
        bad_line = [(2.3, 3.12)]
        tol = 1e-2
        
        line_styles = ['-', '--', ':', '-.']
        style_idx = 0  # index to cycle through styles
        
        for n in n_vals[::skip]:
            for Gamma in Gamma_values[::skip]:
                try:
                    if q == "VeNM":
                        if any(np.isclose(n, bn, atol=tol) and np.isclose(Gamma, bG, atol=tol) for bn, bG in bad_line):
                            continue  
                    # Initial values
                    m0 = 1e-12
                    mp0 = 4*np.pi*a**2*rho0
                    yv0 = np.array([m0, mp0], float)
                    K = p0_rho0/(rho0**(Gamma-1))
                    
                    sol = odeint(f, yv0, rp, args=(n, Gamma, K, Q))
                    mp_sol = sol[:, 0]
                    omega_sol = sol[:, 1]
                    rhor = omega_sol / (4*np.pi*rp**2)
                                   
                    # cs2, q(r), e(r), psi, phi, phi', eps
                    pres = K*rhor**Gamma
                    cs2 = K*Gamma*rhor**(Gamma-1)               
                    q_r = Q*(rp/rb)**n
                    e_r = Q*Q/pow(rb, 2*n)*pow(rp, 2*n-1)/(2*n-1)
                    e_pri = np.gradient(e_r, rp)
                    psidenom = 1-2*mp_sol/rp-e_r/rp+L*rp*rp/3
                    psi = np.log(1/psidenom)/2
                    
                    if not np.all(np.isfinite(psi)):
                        continue
                    
                    phinum = 24*np.pi*K*rhor**Gamma*rp**3+6*mp_sol+3*e_r-3*rp*e_pri+2*L*rp**3
                    phidenom = 2*rp*(3*rp-6*mp_sol-3*e_r+L*rp**3)  
                    phip = rp*(phinum/phidenom)
                    phip1 =(phinum/phidenom)
                    phi = cumulative_trapezoid(phip1, rp, initial=0)
                    
                    if not np.all(np.isfinite(phi)):
                        continue
                    eps = 5
                    e_eps = 6
                    # E(r), F(r), F'(r)
                    E_r = np.exp(phi+psi)*q_r/rp/rp
                    F_r = cumulative_trapezoid(E_r, rp, initial=0)
                    Fpri = np.gradient(F_r, rp)
                    SC = (1-np.exp(-2*psi))/rp/rp
                    Ephi = np.exp(2*phi)
                    Epsi = np.exp(2*psi)
                    
                    #Conditions for SC and Metric
                    crossEphi = np.where(np.diff(np.sign(Ephi - 800)))[0]
                    if len(crossEphi) >= 1:
                        continue
                    crossEpsi = np.where(np.diff(np.sign(Epsi - 800)))[0]
                    if len(crossEpsi) >= 1:
                        continue
                    
                   # Neutral Massless
# =============================================================================
#                     VeffNMs = rp*rp*np.exp(-2*phi)
#                     #Conditions for Neutral Plots
#                     if not np.all(np.isfinite(VeffNMs)):
#                         continue
#                     cross = np.where(np.diff(np.sign(VeffNMs - 1e9)))[0]
#                     if len(cross) >= 1:
#                         continue
#                     if i == 0 and j == 1:
#                         cross1 = np.where(np.diff(np.sign(VeffNMs - 3e7)))[0]
#                         if len(cross1) >= 1:
#                             continue
# =============================================================================

                    # Neutral Massive
                    VeffNMsv = rp*rp*np.exp(-2*phi)-1/eps*rp*rp
                    
                    # Charged Massless
                    VeffChMs = (1+e_eps*F_r)**2*rp*rp*np.exp(-2*phi)
                    #Conditions for Charged Plots
                    if not np.all(np.isfinite(VeffChMs)):
                        continue
                    
                    # Charged Massive
                    VeffChMsv = (1+e_eps*F_r)**2*rp*rp*np.exp(-2*phi)-1/eps/eps*rp*rp
                    #Conditions for Charged Plots
                    if not np.all(np.isfinite(VeffChMsv)):
                        continue
                    
                    if np.all(cs2 < 1):
                        if np.any(rhor < rho0):
                            if not (np.allclose(cs2, 0.25, rtol=1e-3, atol=1e-5)):
                                label = f"n={n:.1f}, Γ={Gamma:.2f}"
                                linestyle = line_styles[style_idx % len(line_styles)]
# =============================================================================
#                                 max_idx_VeNN = argrelextrema(VeffNMs, np.greater, order=10)[0]
#                                 max_idx_VeNM = argrelextrema(VeffNMsv, np.greater, order=10)[0]
#                                 if len(max_idx_VeNN) > 0:
#                                     axs["VeNN"].plot(rp, VeffNMs, lw=1, label=label, linestyle=linestyle)
#                                     axs["VeNN"].legend(loc='best', fontsize=12, frameon=True, facecolor='white', edgecolor='black', framealpha=0.8)
#                                 if len(max_idx_VeNM) > 0:
#                                     axs["VeNM"].plot(rp, VeffNMsv, lw=1, label=label, linestyle=linestyle)
#                                     axs["VeNM"].legend(loc='best', fontsize=12, frameon=True, facecolor='white', edgecolor='black', framealpha=0.8)
# =============================================================================
                                
                                
# =============================================================================
#                                 axs["EPsi"].plot(rp, Epsi, lw=1, label=label, linestyle=linestyle)
#                                 axs["EPhi"].plot(rp, Ephi, lw=1, label=label, linestyle=linestyle)
#                                 axs["SC"].plot(rp, SC, lw=1, label=label, linestyle=linestyle)
#                                 
#                                 axs["EPsi"].legend(loc='best', fontsize=12, frameon=True, facecolor='white', edgecolor='black', framealpha=0.8)
#                                 axs["EPhi"].legend(loc='best', fontsize=12, frameon=True, facecolor='white', edgecolor='black', framealpha=0.8)
#                                 axs["SC"].legend(loc='best', fontsize=12, frameon=True, facecolor='white', edgecolor='black', framealpha=0.8)
# =============================================================================
                                max_idx_VeCN = argrelextrema(VeffChMs, np.greater, order=10)[0]
                                max_idx_VeCM = argrelextrema(VeffChMsv, np.greater, order=10)[0]
                                if len(max_idx_VeCN) > 0:
                                    axs["VeCN"].plot(rp, VeffChMs, lw=1, label=label, linestyle=linestyle)
                                    axs["VeCN"].legend(loc='best', fontsize=12, frameon=True, facecolor='white', edgecolor='black', framealpha=0.8)
                                if len(max_idx_VeCM) > 0:
                                    axs["VeCM"].plot(rp, VeffChMsv, lw=1, label=label, linestyle=linestyle)
                                    axs["VeCM"].legend(loc='best', fontsize=12, frameon=True, facecolor='white', edgecolor='black', framealpha=0.8)
                                
                                style_idx += 1
                except:
                    pass

titles = {
    "VeNN": "$V_{\\rm{eff}}(r)$", "VeNM": "$V_{\\rm{eff}}(r)$", "VeCN": "$V_{\\rm{eff}}(r)$",
    "VeCM": "$V_{\\rm{eff}}(r)$", "EPhi": "e$^{2\\Phi(r)}$", "EPsi": "e$^{2\\Psi(r)}$", "SC": "$k(r)$"
}

for q in plot_quantities:
    fig = figs[q]
    
    for ax_row in np.atleast_2d(axes[q]):
        if isinstance(ax_row, np.ndarray):  # 3x3 grid
            ax_row = np.ravel(ax_row)
        for ax in np.atleast_1d(ax_row):
            label = ax.yaxis.get_label()
            label.set_horizontalalignment('center')
            # Force all rho0 labels to align at same x-position
            ax.yaxis.set_label_coords(-0.17, 0.5) 
            
    # Adjust layout before placing text
    fig.tight_layout()
    fig.subplots_adjust(left=0.13, top=0.9, bottom=0.1, right=0.87, wspace=0.2)

    # Set shared y-axis label
   
    fig.text(0.04, 0.5, titles[q], va='center', rotation='vertical', fontsize=18)

    
    fig.text(0.5, 0.05, "$r$ [m]", ha='center', fontsize=18)
    filename = f"{q}_plot.pdf"
    plt.minorticks_on()
    fig.savefig(filename, format="pdf", bbox_inches='tight')
    #fig.show()
#%%
# Scatter Code
import numpy as np
from scipy.integrate import odeint, cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = '12'
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.linewidth'] = 1.
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.handletextpad'] = 0.3

# Constants
p0_rho0 = 0.25
rb, L = 1e4, 1e-52
rho0_list = [2e-9, 5e-9, 1e-8]
Q_list = [1e3, 3e3, 5e3]
a = 0.01
b = rb
N = 15000
rp = np.linspace(a, b, N)

Gamma_values = np.linspace(1, 5, 100)
n_vals = np.arange(1, 5.1, 0.01)

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
    fo = t1-t2+t3
    return np.array([ft, fo], float)

# Storage for scatter data
scatter_data = {}
skip= 1
for rho0 in rho0_list:
    for Q in Q_list:
        valid_points = []

        for n in n_vals[::skip]:
            for Gamma in Gamma_values[::skip]:
                try:
                    m0 = 1e-12
                    mp0 = 4*np.pi*a**2*rho0
                    yv0 = np.array([m0, mp0], float)
                    K = p0_rho0 / (rho0**(Gamma - 1))
                    sol = odeint(f, yv0, rp, args=(n, Gamma, K, Q))
                    mp_sol, omega_sol = sol[:, 0], sol[:, 1]
        
                    rhor = omega_sol / (4 * np.pi * rp**2)
                    pres = K * rhor**Gamma
                    cs2 = K * Gamma * rhor**(Gamma - 1)
        
                    # Electromagnetic terms
                    q_r = Q * (rp / rb)**n
                    e_r = Q**2 / rb**(2*n) * rp**(2*n-1) / (2*n - 1)
                    e_pri = np.gradient(e_r, rp)
        
                    # Metric potentials
                    phinum = 24*np.pi*K*rhor**Gamma*rp**3 + 6*mp_sol + 3*e_r - 3*rp*e_pri + 2*L*rp**3
                    phidenom = 2*rp*(3*rp - 6*mp_sol - 3*e_r + L*rp**3)
                    phip = phinum / phidenom
                    phi = cumulative_trapezoid(phip, rp, initial=0)
        
                    psidenom = 1 - 2*mp_sol/rp - e_r/rp + L*rp**2/3
                    psi = np.log(1/psidenom) / 2
        
                    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(psi)):
                        continue
        
                    # Compute E, F
                    eps2 = 5
                    e_eps = 8.3
                    E_r = np.exp(phi + psi) * q_r / (rp**2)
                    F_r = cumulative_trapezoid(E_r, rp, initial=0)
                    
                    #VeffNMs = rp**2 * np.exp(-2 * phi) 
# =============================================================================
#                     if not np.all(np.isfinite(VeffNMs)):
#                         continue
# =============================================================================
                                
# =============================================================================
#                     VeffNMsv = rp*rp*np.exp(-2*phi)-1/eps2*rp*rp
#                     if not np.all(np.isfinite(VeffNMsv)):
#                         continue
# =============================================================================
                    
# =============================================================================
#                     VeffChMs = (1+e_eps*F_r)**2*rp*rp*np.exp(-2*phi)
#                     if not np.all(np.isfinite(VeffChMs)):
#                         continue
# =============================================================================
                    
                    VeffChMsv = (1+e_eps*F_r)**2*rp*rp*np.exp(-2*phi)-1/eps/eps*rp*rp
                    if not np.all(np.isfinite(VeffChMsv)):
                        continue
                    
        
                    # Check for local maxima
                    from scipy.signal import argrelextrema
                    #max_idx_VeNN = argrelextrema(VeffNMs, np.greater, order=10)[0] 
                    #max_idx_VeNM = argrelextrema(VeffNMsv, np.greater, order=10)[0] 
                    #max_idx_VeCN = argrelextrema(VeffChMs, np.greater, order=10)[0] 
                    max_idx_VeCM = argrelextrema(VeffChMsv, np.greater, order=10)[0]
        
                    if np.all(cs2 < 1) and np.any(rhor < rho0):
                        #valid_points.append((n, Gamma))
                        #if len(max_idx_VeNN) > 0:
                        #if len(max_idx_VeNM) > 0:
                        #if len(max_idx_VeCN) > 0:
                            #valid_points.append((n, Gamma))
                        if len(max_idx_VeCM) > 0:
                            valid_points.append((n, Gamma))
        
                except Exception:
                    continue

        scatter_data[(rho0, Q)] = np.array(valid_points)

# Plot all scatter results
fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
for i, rho0 in enumerate(rho0_list):
    for j, Q in enumerate(Q_list):
        ax = axes[i, j]
        pts = scatter_data.get((rho0, Q), np.empty((0, 2)))
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=20, color='green', rasterized=True)

        rho_label = f"{rho0:.0e}"
        mantissa, exp = rho_label.split("e")
        exp = int(exp) 
        rho_str = rf"{mantissa}\times 10^{{{exp}}}"
        
        Q_label = f"{Q:.0e}"
        mantissa1, exp1 = Q_label.split("e")
        exp1 = int(exp1)
        Q_str = rf"{mantissa1}\times 10^{{{exp1}}}"
        
        titlerho = f"$\\rho_0 = {rho_str}$"
        titleQ= f"$Q = {Q_str}$"
        
        if j == 0:
            ax.set_ylabel(titlerho, fontsize = 16)
        if i == 0:
            ax.set_title(titleQ, rotation = 'horizontal', fontsize = 16)
        ax.grid(True)
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
        ax.tick_params(direction='in')
        ax.set_xlim(0.8, 5.2)


fig.tight_layout()
fig.subplots_adjust(left=0.09, top=0.95, bottom=0.06)
fig.text(0.02, 0.5, "$n$", va='center', rotation='vertical', fontsize=18)
fig.text(0.5, 0.01, "$\\Gamma$", ha='center', fontsize=18)
fig.savefig("CM_10_scatter_skip5.pdf", format="pdf", bbox_inches="tight")
plt.show()