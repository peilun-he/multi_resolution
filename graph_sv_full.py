"""
Graph Signal Estimation for Multivariate Stochastic Volatility
via Warped Graph Gaussian Processes

Single-resolution: d=10 assets, RBF kernel, exp warping
Methods: ICM/Newton MAP + Laplace approximation + Expectation Propagation
Includes: forecasting under both approximations, comparison diagnostics

Authors: Peters, Campi, He, Zhu
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvalsh, solve
from scipy.special import roots_hermite
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from time import time

np.random.seed(42)

# ============================================================
# 1. MODEL SETUP
# ============================================================

d = 10       # assets
T = 200      # training periods
h_fore = 20  # forecast horizon

# --- RBF temporal kernel ---
def rbf_kernel(t1, t2, length_scale=20.0, variance=1.0):
    t1 = np.asarray(t1).reshape(-1, 1)
    t2 = np.asarray(t2).reshape(1, -1)
    return variance * np.exp(-0.5 * (t1 - t2)**2 / length_scale**2)

times_train = np.arange(1, T + 1, dtype=float)
times_fore  = np.arange(T + 1, T + 1 + h_fore, dtype=float)

K_tt = rbf_kernel(times_train, times_train) + 1e-5 * np.eye(T)
K_ts = rbf_kernel(times_train, times_fore)
K_ss = rbf_kernel(times_fore, times_fore)  + 1e-5 * np.eye(h_fore)
K_tt_cho = cho_factor(K_tt)
K_t_inv  = cho_solve(K_tt_cho, np.eye(T))

# --- Asset graph ---
def make_asset_graph(d, p_edge=0.4, seed=123):
    rng = np.random.RandomState(seed)
    W = np.zeros((d, d))
    perm = rng.permutation(d)
    for k in range(d - 1):
        w = rng.uniform(0.3, 1.0)
        W[perm[k], perm[k+1]] = w
        W[perm[k+1], perm[k]] = w
    for i in range(d):
        for j in range(i+1, d):
            if W[i,j] == 0 and rng.rand() < p_edge:
                w = rng.uniform(0.1, 1.0)
                W[i,j] = w; W[j,i] = w
    D = np.diag(W.sum(axis=1))
    return D - W, W

L_graph, W_graph = make_asset_graph(d)
alpha_g, beta_g = 1.5, 1.0
Q_g     = alpha_g * L_graph + beta_g * np.eye(d)
Q_g_inv = np.linalg.inv(Q_g)

# Full prior precision Q = K_t^{-1} ⊗ Q_g  (N x N, N = dT)
N = d * T
Q_full = np.kron(K_t_inv, Q_g)

# ============================================================
# 2. SIMULATE DATA
# ============================================================
print("=" * 70)
print("1. SIMULATING DATA")
print("=" * 70)

L_Qg = np.linalg.cholesky(Q_g_inv)
L_Kt = np.linalg.cholesky(K_tt)

H_true = L_Qg @ np.random.randn(d, T) @ L_Kt.T   # d x T
S_true = H_true.copy()
V_true = np.exp(S_true)

mu = np.zeros(d)
R  = mu[:, None] + np.sqrt(V_true) * np.random.randn(d, T)

# Future truth for evaluation
A_time = cho_solve(K_tt_cho, K_ts)          # T x h
K_cond = K_ss - K_ts.T @ A_time             # h x h
L_cond = np.linalg.cholesky(K_cond + 1e-7 * np.eye(h_fore))
H_future_true = H_true @ A_time + L_Qg @ np.random.randn(d, h_fore) @ L_cond.T
S_future_true = H_future_true.copy()
V_future_true = np.exp(S_future_true)

print(f"  d={d}, T={T}, h_fore={h_fore}")
print(f"  Log-vol range:  [{S_true.min():.2f}, {S_true.max():.2f}]")
print(f"  Vol range:      [{V_true.min():.4f}, {V_true.max():.4f}]")

# ============================================================
# 3. LIKELIHOOD FUNCTIONS
# ============================================================

def ll_val(r, s):
    return -0.5 * (np.log(2*np.pi) + s + r**2 * np.exp(-s))

def ll_d1(r, s):
    return 0.5 * (r**2 * np.exp(-s) - 1.0)

def ll_d2(r, s):
    return -0.5 * r**2 * np.exp(-s)

# ============================================================
# 4. MAP ESTIMATION (ICM / Newton)
# ============================================================
print("\n" + "=" * 70)
print("2. MAP ESTIMATION (ICM / Newton)")
print("=" * 70)

t0 = time()

# Initialise: smoothed log(r^2)
def smooth_1d(x, w=20):
    k = np.ones(w)/w
    p = np.pad(x, (w//2, w//2), mode='edge')
    return np.convolve(p, k, mode='valid')[:len(x)]

log_r2 = np.log(R**2 + 0.01)
H_est = np.zeros((d, T))
for i in range(d):
    H_est[i] = smooth_1d(log_r2[i])
cross_mean = H_est.mean(axis=0, keepdims=True)
H_est = 0.6 * H_est + 0.4 * cross_mean

def log_posterior(H):
    return np.sum(ll_val(R, H)) - 0.5 * np.trace(Q_g @ H @ K_t_inv @ H.T)

obj_hist_map = []
max_icm = 80; tol_icm = 1e-7

for it in range(max_icm):
    H_old = H_est.copy()
    for t in range(T):
        q_tt = K_t_inv[t, t]
        Q_tt = q_tt * Q_g
        rhs = sum(K_t_inv[t, tp] * Q_g @ H_est[:, tp] for tp in range(T) if tp != t)
        mu_c = -solve(Q_tt, rhs, assume_a='pos')

        u = H_est[:, t].copy()
        for _ in range(10):
            g = ll_d1(R[:, t], u)
            hd = ll_d2(R[:, t], u)
            grad = g - Q_tt @ (u - mu_c)
            Hn = Q_tt + np.diag(-hd)
            delta = solve(Hn, grad, assume_a='pos')
            # Backtracking
            obj_c = np.sum(ll_val(R[:, t], u)) - 0.5*(u-mu_c)@Q_tt@(u-mu_c)
            step = 1.0
            for _ in range(12):
                un = u + step * delta
                obj_n = np.sum(ll_val(R[:, t], un)) - 0.5*(un-mu_c)@Q_tt@(un-mu_c)
                if obj_n > obj_c - 1e-4 * step * grad @ delta:
                    break
                step *= 0.5
            u += step * delta
            if np.linalg.norm(step * delta) < 1e-10:
                break
        H_est[:, t] = u

    obj = log_posterior(H_est)
    obj_hist_map.append(obj)
    rc = np.linalg.norm(H_est - H_old) / (np.linalg.norm(H_old) + 1e-10)
    if it % 10 == 0:
        print(f"  iter {it+1:3d}: log-post={obj:.1f}, rel_change={rc:.2e}")
    if rc < tol_icm:
        print(f"  Converged at iter {it+1}")
        break

H_map = H_est.copy()
S_map = H_map.copy()
t_map = time() - t0
print(f"  MAP time: {t_map:.1f}s")

# ============================================================
# 5. LAPLACE APPROXIMATION
# ============================================================
print("\n" + "=" * 70)
print("3. LAPLACE APPROXIMATION")
print("=" * 70)

t0 = time()
W_lap = np.zeros(N)
for t in range(T):
    for i in range(d):
        W_lap[t*d + i] = -ll_d2(R[i, t], S_map[i, t])

Prec_lap = Q_full + np.diag(W_lap)
Sigma_lap = np.linalg.inv(Prec_lap)
marg_var_lap = np.diag(Sigma_lap).reshape(T, d).T  # d x T

t_lap = time() - t0
print(f"  Posterior std range: [{np.sqrt(marg_var_lap.min()):.4f}, {np.sqrt(marg_var_lap.max()):.4f}]")
print(f"  Laplace time: {t_lap:.1f}s")

# ============================================================
# 6. EXPECTATION PROPAGATION
# ============================================================
print("\n" + "=" * 70)
print("4. EXPECTATION PROPAGATION")
print("=" * 70)

t0 = time()

# EP sites: for each (i,t), scalar natural parameters (tau_tilde, nu_tilde)
# since single resolution m=1, block sites reduce to scalar sites
# Global posterior: q(h) = N(m, Sigma)
#   Sigma^{-1} = Q + diag(tau_tilde)
#   m = Sigma * nu_tilde

tau_tilde = np.zeros(N)   # site precisions
nu_tilde  = np.zeros(N)   # site natural means

# Gauss-Hermite quadrature nodes/weights
n_gh = 30
gh_x, gh_w = np.polynomial.hermite.hermgauss(n_gh)
# Transform: if z ~ N(mu, sig^2), then z = mu + sqrt(2)*sig*x_gh
# and E[f(z)] ≈ (1/sqrt(pi)) * sum_k w_k * f(mu + sqrt(2)*sig*x_k)

def ep_moments_tilted(r_it, mu_i, mu_cav, var_cav):
    """
    Compute moments of tilted distribution:
      p_tilt(s) ∝ N(s | mu_cav, var_cav) * p(r_it | s)
    
    Returns: log Z_tilt, E[s], E[s^2] under tilted
    """
    sig_cav = np.sqrt(var_cav)
    # Quadrature points in s-space
    s_pts = mu_cav + np.sqrt(2) * sig_cav * gh_x   # (n_gh,)
    
    # Log-likelihood at quadrature points
    log_lik = ll_val(r_it, s_pts)   # (n_gh,)
    
    # For numerical stability, subtract max
    log_lik_max = np.max(log_lik)
    w_eff = gh_w * np.exp(log_lik - log_lik_max)   # (n_gh,)
    
    Z = np.sum(w_eff) / np.sqrt(np.pi)
    if Z < 1e-30:
        return -1e10, mu_cav, var_cav
    
    log_Z = np.log(Z) + log_lik_max
    
    # Moments
    E_s   = np.sum(w_eff * s_pts) / (np.sqrt(np.pi) * Z)
    E_s2  = np.sum(w_eff * s_pts**2) / (np.sqrt(np.pi) * Z)
    var_s = E_s2 - E_s**2
    
    # Protect against negative variance
    var_s = max(var_s, 1e-10)
    
    return log_Z, E_s, var_s


# --- EP main loop ---
max_ep_sweeps = 60
damp = 0.8        # damping factor (1 = no damping)
tol_ep = 1e-6

# Initialise global posterior from prior
# Start with tau_tilde = 0 → Sigma = Q^{-1}, m = 0
# But we can warm-start from Laplace sites
for t in range(T):
    for i in range(d):
        n = t * d + i
        tau_tilde[n] = W_lap[n]    # Laplace curvature as init
        nu_tilde[n] = W_lap[n] * S_map[i, t]  # centered at MAP

Prec_ep = Q_full + np.diag(tau_tilde)
Sigma_ep = np.linalg.inv(Prec_ep)
m_ep = Sigma_ep @ nu_tilde

ep_log_Z_hist = []

print(f"  Running EP with {max_ep_sweeps} sweeps, damping={damp}, GH nodes={n_gh}")

for sweep in range(max_ep_sweeps):
    max_delta_tau = 0.0
    sum_log_Z = 0.0
    
    for t in range(T):
        for i in range(d):
            n = t * d + i
            
            # Current marginal
            mu_n  = m_ep[n]
            var_n = Sigma_ep[n, n]
            
            # Cavity: remove site
            tau_cav = 1.0 / var_n - tau_tilde[n]
            if tau_cav < 1e-10:
                tau_cav = 1e-10
            var_cav = 1.0 / tau_cav
            nu_cav  = mu_n / var_n - nu_tilde[n]
            mu_cav  = var_cav * nu_cav
            
            # Tilted moments via GH quadrature
            log_Z_t, mu_tilt, var_tilt = ep_moments_tilted(
                R[i, t], mu[i], mu_cav, var_cav
            )
            sum_log_Z += log_Z_t
            
            # New site from moment matching
            tau_new = 1.0 / var_tilt - tau_cav
            nu_new  = mu_tilt / var_tilt - nu_cav
            
            # Enforce non-negative precision
            if tau_new < 0:
                tau_new = 1e-8
                nu_new = nu_tilde[n]  # keep old
            
            # Damped update
            tau_upd = (1 - damp) * tau_tilde[n] + damp * tau_new
            nu_upd  = (1 - damp) * nu_tilde[n]  + damp * nu_new
            
            max_delta_tau = max(max_delta_tau, abs(tau_upd - tau_tilde[n]))
            
            tau_tilde[n] = tau_upd
            nu_tilde[n]  = nu_upd
    
    # Recompute global posterior
    Prec_ep = Q_full + np.diag(tau_tilde)
    
    # Check positive definiteness via Cholesky
    try:
        Sigma_ep = np.linalg.inv(Prec_ep)
        m_ep = Sigma_ep @ nu_tilde
    except np.linalg.LinAlgError:
        print(f"  WARNING: Prec_ep not PD at sweep {sweep+1}, reducing damping")
        damp *= 0.5
        continue
    
    ep_log_Z_hist.append(sum_log_Z)
    
    if sweep % 5 == 0 or max_delta_tau < tol_ep:
        print(f"  sweep {sweep+1:3d}: max_delta_tau={max_delta_tau:.2e}, "
              f"sum_logZ={sum_log_Z:.1f}")
    
    if max_delta_tau < tol_ep:
        print(f"  EP converged at sweep {sweep+1}")
        break

# Extract EP marginals
marg_var_ep = np.diag(Sigma_ep).reshape(T, d).T   # d x T
S_ep = m_ep.reshape(T, d).T                        # d x T

t_ep = time() - t0
print(f"  EP posterior mean range: [{S_ep.min():.3f}, {S_ep.max():.3f}]")
print(f"  EP posterior std range:  [{np.sqrt(marg_var_ep.min()):.4f}, {np.sqrt(marg_var_ep.max()):.4f}]")
print(f"  EP time: {t_ep:.1f}s")

# ============================================================
# 7. FORECASTING (both Laplace and EP)
# ============================================================
print("\n" + "=" * 70)
print("5. FORECASTING")
print("=" * 70)

def forecast(m_train_mat, Sigma_train):
    """
    Compute predictive Gaussian for h_* given posterior q(h) = N(m, Sigma).
    m_train_mat: d x T posterior mean as matrix
    Sigma_train: (dT x dT) posterior covariance
    
    Returns: forecast_mean_s (d x h), forecast_var_s (d x h)
    """
    A = cho_solve(K_tt_cho, K_ts)    # T x h
    K_cd = K_ss - K_ts.T @ A         # h x h
    
    # Predictive mean: M_* = M_train @ A
    M_star = m_train_mat @ A          # d x h
    
    # Predictive covariance: Σ_* = K_cd ⊗ Q_g^{-1} + (A^T ⊗ I) Σ (A ⊗ I)
    term1 = np.kron(K_cd, Q_g_inv)
    AkI = np.kron(A.T, np.eye(d))
    term2 = AkI @ Sigma_train @ AkI.T
    Sig_star = term1 + term2
    
    # Extract marginals
    m_vec = M_star.T.reshape(-1)
    fmean = np.zeros((d, h_fore))
    fvar  = np.zeros((d, h_fore))
    for tf in range(h_fore):
        for i in range(d):
            idx = tf * d + i
            fmean[i, tf] = m_vec[idx]
            fvar[i, tf]  = max(Sig_star[idx, idx], 1e-10)
    
    return fmean, fvar, Sig_star

# Laplace forecasts
fmean_lap, fvar_lap, Sig_star_lap = forecast(H_map, Sigma_lap)
fmean_v_lap = np.exp(fmean_lap + 0.5 * fvar_lap)
fvar_v_lap  = (np.exp(fvar_lap) - 1) * np.exp(2*fmean_lap + fvar_lap)

# EP forecasts
fmean_ep, fvar_ep, Sig_star_ep = forecast(S_ep, Sigma_ep)
fmean_v_ep = np.exp(fmean_ep + 0.5 * fvar_ep)
fvar_v_ep  = (np.exp(fvar_ep) - 1) * np.exp(2*fmean_ep + fvar_ep)

print(f"  Laplace forecast log-vol std range: [{np.sqrt(fvar_lap.min()):.3f}, {np.sqrt(fvar_lap.max()):.3f}]")
print(f"  EP forecast log-vol std range:      [{np.sqrt(fvar_ep.min()):.3f}, {np.sqrt(fvar_ep.max()):.3f}]")

# ============================================================
# 8. EVALUATION METRICS
# ============================================================
print("\n" + "=" * 70)
print("6. EVALUATION METRICS")
print("=" * 70)

def eval_metrics(S_est, marg_var, label, S_ref=S_true):
    rmse = np.sqrt(np.mean((S_ref - S_est)**2))
    corr = np.corrcoef(S_ref.ravel(), S_est.ravel())[0, 1]
    mae  = np.mean(np.abs(S_ref - S_est))
    
    # CI coverage
    in_ci = sum(1 for t in range(T) for i in range(d)
                if abs(S_ref[i,t] - S_est[i,t]) <= 1.96 * np.sqrt(marg_var[i,t]))
    cov = in_ci / (d * T)
    
    # Average CI width
    avg_width = np.mean(2 * 1.96 * np.sqrt(marg_var))
    
    print(f"  [{label}] RMSE={rmse:.4f}, MAE={mae:.4f}, Corr={corr:.4f}, "
          f"95% Coverage={cov:.3f}, Avg CI width={avg_width:.3f}")
    return rmse, corr, cov, avg_width

rmse_lap, corr_lap, cov_lap, w_lap = eval_metrics(S_map, marg_var_lap, "Laplace")
rmse_ep,  corr_ep,  cov_ep,  w_ep  = eval_metrics(S_ep,  marg_var_ep,  "EP")

# Forecast coverage
def forecast_coverage(fmean, fvar, S_ref):
    in_ci = sum(1 for t in range(h_fore) for i in range(d)
                if abs(S_ref[i,t] - fmean[i,t]) <= 1.96 * np.sqrt(fvar[i,t]))
    return in_ci / (d * h_fore)

fcov_lap = forecast_coverage(fmean_lap, fvar_lap, S_future_true)
fcov_ep  = forecast_coverage(fmean_ep,  fvar_ep,  S_future_true)
print(f"  [Laplace] Forecast 95% coverage: {fcov_lap:.3f}")
print(f"  [EP]      Forecast 95% coverage: {fcov_ep:.3f}")

# Per-asset metrics
print(f"\n  Per-asset RMSE / Correlation:")
rmse_asset_lap = np.sqrt(np.mean((S_true - S_map)**2, axis=1))
rmse_asset_ep  = np.sqrt(np.mean((S_true - S_ep)**2, axis=1))
corr_asset_lap = [np.corrcoef(S_true[i], S_map[i])[0,1] for i in range(d)]
corr_asset_ep  = [np.corrcoef(S_true[i], S_ep[i])[0,1] for i in range(d)]
for i in range(d):
    print(f"    Asset {i}: Laplace RMSE={rmse_asset_lap[i]:.3f} r={corr_asset_lap[i]:.3f} | "
          f"EP RMSE={rmse_asset_ep[i]:.3f} r={corr_asset_ep[i]:.3f}")

# ============================================================
# 9. COMPREHENSIVE PLOTS
# ============================================================
print("\n" + "=" * 70)
print("7. GENERATING PLOTS")
print("=" * 70)

assets_show = [0, 3, 7]
colors_asset = plt.cm.tab10(np.linspace(0, 1, 10))

# ---- FIGURE 1: Model setup & graph structure ----
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
im = ax.imshow(W_graph, cmap='Blues', interpolation='nearest')
ax.set_title('Asset Graph Adjacency', fontweight='bold')
ax.set_xlabel('Asset'); ax.set_ylabel('Asset')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1]
eigs_L = eigvalsh(L_graph); eigs_Q = eigvalsh(Q_g)
ax.bar(np.arange(d)-0.15, eigs_L, 0.3, label='$L$', color='steelblue', alpha=0.7)
ax.bar(np.arange(d)+0.15, eigs_Q, 0.3, label='$Q_g$', color='coral', alpha=0.7)
ax.set_title('Graph Spectral Properties', fontweight='bold')
ax.set_xlabel('Index'); ax.legend(); ax.set_xticks(range(d))

ax = axes[2]
ai = 0
ax.plot(times_train, R[ai], color='gray', alpha=0.4, lw=0.5, label='Returns')
ax.plot(times_train,  2*np.sqrt(V_true[ai]), 'k-', lw=1, label='$\\pm 2\\sigma$ true')
ax.plot(times_train, -2*np.sqrt(V_true[ai]), 'k-', lw=1)
ax.set_title(f'Asset {ai}: Returns & True Vol Envelope', fontweight='bold')
ax.set_xlabel('Time'); ax.legend(fontsize=8)

fig1.tight_layout()
fig1.savefig('/home/claude/fig1_setup.png', dpi=150, bbox_inches='tight')
plt.close(fig1)

# ---- FIGURE 2: Laplace vs EP estimation (3 assets) ----
fig2, axes = plt.subplots(3, 2, figsize=(18, 14))

for row, ai in enumerate(assets_show):
    # Log-volatility
    ax = axes[row, 0]
    # Laplace
    std_l = np.sqrt(marg_var_lap[ai])
    ax.fill_between(times_train, S_map[ai]-1.96*std_l, S_map[ai]+1.96*std_l,
                     alpha=0.15, color='steelblue', label='Laplace 95% CI')
    ax.plot(times_train, S_map[ai], '-', color='steelblue', lw=1.2, label='Laplace MAP')
    # EP
    std_e = np.sqrt(marg_var_ep[ai])
    ax.fill_between(times_train, S_ep[ai]-1.96*std_e, S_ep[ai]+1.96*std_e,
                     alpha=0.15, color='coral', label='EP 95% CI')
    ax.plot(times_train, S_ep[ai], '-', color='coral', lw=1.2, label='EP mean')
    # Truth
    ax.plot(times_train, S_true[ai], 'k-', alpha=0.5, lw=0.8, label='True')
    ax.set_title(f'Asset {ai}: Log-Volatility $h_{{i,t}}$', fontweight='bold')
    ax.set_ylabel('$h_{i,t}$')
    if row == 0:
        ax.legend(fontsize=7, ncol=2)
    if row == 2:
        ax.set_xlabel('Time')

    # Volatility
    ax = axes[row, 1]
    ax.plot(times_train, V_true[ai], 'k-', alpha=0.5, lw=0.8, label='True')
    ax.plot(times_train, np.exp(S_map[ai]), '-', color='steelblue', lw=1.2, label='Laplace')
    ax.plot(times_train, np.exp(S_ep[ai]), '-', color='coral', lw=1.2, label='EP')
    ax.set_title(f'Asset {ai}: Volatility $v_{{i,t}}$', fontweight='bold')
    ax.set_ylabel('$v_{i,t}$')
    if row == 0:
        ax.legend(fontsize=8)
    if row == 2:
        ax.set_xlabel('Time')

fig2.suptitle('Laplace vs EP: Estimation Comparison', fontsize=14, fontweight='bold', y=1.01)
fig2.tight_layout()
fig2.savefig('/home/claude/fig2_estimation.png', dpi=150, bbox_inches='tight')
plt.close(fig2)

# ---- FIGURE 3: Forecasting comparison ----
fig3, axes = plt.subplots(3, 2, figsize=(18, 14))

for row, ai in enumerate(assets_show):
    # Log-vol forecast
    ax = axes[row, 0]
    # Last bit of training
    t_show = 50
    ax.plot(times_train[-t_show:], S_true[ai, -t_show:], 'k-', alpha=0.5, lw=0.8, label='True')
    ax.plot(times_train[-t_show:], S_map[ai, -t_show:], '-', color='steelblue', lw=1, alpha=0.7)
    ax.plot(times_train[-t_show:], S_ep[ai, -t_show:], '-', color='coral', lw=1, alpha=0.7)
    
    # Laplace forecast
    fstd_l = np.sqrt(fvar_lap[ai])
    ax.fill_between(times_fore, fmean_lap[ai]-1.96*fstd_l, fmean_lap[ai]+1.96*fstd_l,
                     alpha=0.15, color='steelblue', label='Laplace 95% CI')
    ax.plot(times_fore, fmean_lap[ai], '-', color='steelblue', lw=1.5, label='Laplace')
    
    # EP forecast
    fstd_e = np.sqrt(fvar_ep[ai])
    ax.fill_between(times_fore, fmean_ep[ai]-1.96*fstd_e, fmean_ep[ai]+1.96*fstd_e,
                     alpha=0.15, color='coral', label='EP 95% CI')
    ax.plot(times_fore, fmean_ep[ai], '-', color='coral', lw=1.5, label='EP')
    
    ax.plot(times_fore, S_future_true[ai], 'k--', alpha=0.5, lw=0.8, label='True (future)')
    ax.axvline(T+0.5, color='gray', ls=':', lw=0.8)
    ax.set_title(f'Asset {ai}: Log-Vol Forecast', fontweight='bold')
    ax.set_ylabel('$h_{i,t}$')
    if row == 0:
        ax.legend(fontsize=7, ncol=2)
    if row == 2:
        ax.set_xlabel('Time')

    # Vol forecast
    ax = axes[row, 1]
    ax.plot(times_train[-t_show:], V_true[ai, -t_show:], 'k-', alpha=0.5, lw=0.8, label='True')
    
    fstd_vl = np.sqrt(fvar_v_lap[ai])
    ax.fill_between(times_fore, np.maximum(fmean_v_lap[ai]-1.96*fstd_vl, 0),
                     fmean_v_lap[ai]+1.96*fstd_vl,
                     alpha=0.12, color='steelblue', label='Laplace CI')
    ax.plot(times_fore, fmean_v_lap[ai], '-', color='steelblue', lw=1.5, label='Laplace')
    
    fstd_ve = np.sqrt(fvar_v_ep[ai])
    ax.fill_between(times_fore, np.maximum(fmean_v_ep[ai]-1.96*fstd_ve, 0),
                     fmean_v_ep[ai]+1.96*fstd_ve,
                     alpha=0.12, color='coral', label='EP CI')
    ax.plot(times_fore, fmean_v_ep[ai], '-', color='coral', lw=1.5, label='EP')
    
    ax.plot(times_fore, V_future_true[ai], 'k--', alpha=0.5, lw=0.8, label='True (future)')
    ax.axvline(T+0.5, color='gray', ls=':', lw=0.8)
    ax.set_title(f'Asset {ai}: Volatility Forecast', fontweight='bold')
    ax.set_ylabel('$v_{i,t}$')
    if row == 0:
        ax.legend(fontsize=7, ncol=2)
    if row == 2:
        ax.set_xlabel('Time')

fig3.suptitle('Laplace vs EP: Forecasting Comparison', fontsize=14, fontweight='bold', y=1.01)
fig3.tight_layout()
fig3.savefig('/home/claude/fig3_forecasting.png', dpi=150, bbox_inches='tight')
plt.close(fig3)

# ---- FIGURE 4: Heatmaps and diagnostics ----
fig4 = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

vmin, vmax = S_true.min(), S_true.max()

ax = fig4.add_subplot(gs[0, 0])
ax.imshow(S_true, aspect='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
ax.set_title('True $h_{i,t}$', fontweight='bold')
ax.set_xlabel('Time'); ax.set_ylabel('Asset')

ax = fig4.add_subplot(gs[0, 1])
ax.imshow(S_map, aspect='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
ax.set_title('Laplace MAP $h_{i,t}$', fontweight='bold')
ax.set_xlabel('Time'); ax.set_ylabel('Asset')

ax = fig4.add_subplot(gs[0, 2])
ax.imshow(S_ep, aspect='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
ax.set_title('EP Mean $h_{i,t}$', fontweight='bold')
ax.set_xlabel('Time'); ax.set_ylabel('Asset')

# Posterior std heatmaps
ax = fig4.add_subplot(gs[1, 0])
im = ax.imshow(np.sqrt(marg_var_lap), aspect='auto', cmap='viridis')
ax.set_title('Laplace Posterior Std', fontweight='bold')
ax.set_xlabel('Time'); ax.set_ylabel('Asset')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = fig4.add_subplot(gs[1, 1])
im = ax.imshow(np.sqrt(marg_var_ep), aspect='auto', cmap='viridis')
ax.set_title('EP Posterior Std', fontweight='bold')
ax.set_xlabel('Time'); ax.set_ylabel('Asset')
plt.colorbar(im, ax=ax, fraction=0.046)

# Ratio of stds
ax = fig4.add_subplot(gs[1, 2])
ratio = np.sqrt(marg_var_ep) / np.sqrt(marg_var_lap)
im = ax.imshow(ratio, aspect='auto', cmap='RdBu_r',
               vmin=max(0.5, ratio.min()), vmax=min(3.0, ratio.max()))
ax.set_title('Std Ratio (EP / Laplace)', fontweight='bold')
ax.set_xlabel('Time'); ax.set_ylabel('Asset')
plt.colorbar(im, ax=ax, fraction=0.046)

# Per-asset metrics
ax = fig4.add_subplot(gs[2, 0])
x_pos = np.arange(d)
ax.bar(x_pos-0.15, rmse_asset_lap, 0.3, label='Laplace', color='steelblue', alpha=0.7)
ax.bar(x_pos+0.15, rmse_asset_ep,  0.3, label='EP', color='coral', alpha=0.7)
ax.set_title('RMSE per Asset', fontweight='bold')
ax.set_xlabel('Asset'); ax.set_ylabel('RMSE')
ax.legend(); ax.set_xticks(range(d))

ax = fig4.add_subplot(gs[2, 1])
ax.bar(x_pos-0.15, corr_asset_lap, 0.3, label='Laplace', color='steelblue', alpha=0.7)
ax.bar(x_pos+0.15, corr_asset_ep,  0.3, label='EP', color='coral', alpha=0.7)
ax.set_title('Correlation per Asset', fontweight='bold')
ax.set_xlabel('Asset'); ax.set_ylabel('Corr')
ax.legend(); ax.set_xticks(range(d))

# Convergence
ax = fig4.add_subplot(gs[2, 2])
ax.plot(obj_hist_map, 'o-', color='steelblue', ms=2, label='MAP log-posterior')
ax.set_title('ICM Convergence', fontweight='bold')
ax.set_xlabel('Iteration'); ax.set_ylabel('Log-posterior')
ax.legend()

fig4.savefig('/home/claude/fig4_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close(fig4)

# ---- FIGURE 5: Summary comparison panel ----
fig5, axes = plt.subplots(1, 3, figsize=(18, 5))

# Coverage bar chart
ax = axes[0]
methods = ['Laplace\n(train)', 'EP\n(train)', 'Laplace\n(forecast)', 'EP\n(forecast)']
coverages = [cov_lap, cov_ep, fcov_lap, fcov_ep]
bar_colors = ['steelblue','coral','#7ab0d4','#f0a08a']
bars = ax.bar(methods, coverages, color=bar_colors, alpha=0.8)
ax.axhline(0.95, color='k', ls='--', lw=1, label='Nominal 95%')
ax.set_title('95% CI Coverage', fontweight='bold')
ax.set_ylabel('Coverage')
ax.legend()
ax.set_ylim(0, 1.05)
for bar, v in zip(bars, coverages):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{v:.2f}', ha='center', fontsize=10)

# CI width comparison
ax = axes[1]
ax.plot(times_train, 2*1.96*np.sqrt(marg_var_lap).mean(axis=0),
        color='steelblue', lw=1.5, label='Laplace avg width')
ax.plot(times_train, 2*1.96*np.sqrt(marg_var_ep).mean(axis=0),
        color='coral', lw=1.5, label='EP avg width')
ax.set_title('Average 95% CI Width Over Time', fontweight='bold')
ax.set_xlabel('Time'); ax.set_ylabel('Width')
ax.legend()

# Scatter: Laplace vs EP means
ax = axes[2]
ax.scatter(S_map.ravel(), S_ep.ravel(), alpha=0.1, s=5, color='gray')
lims = [min(S_map.min(), S_ep.min())-0.5, max(S_map.max(), S_ep.max())+0.5]
ax.plot(lims, lims, 'k--', lw=0.8)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title('Laplace MAP vs EP Mean', fontweight='bold')
ax.set_xlabel('Laplace MAP $h_{i,t}$')
ax.set_ylabel('EP Mean $h_{i,t}$')
ax.set_aspect('equal')

fig5.tight_layout()
fig5.savefig('/home/claude/fig5_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig5)

print("\n  All figures saved.")
print(f"\n  Timing: MAP={t_map:.1f}s, Laplace={t_lap:.1f}s, EP={t_ep:.1f}s")
print("  Done.")
