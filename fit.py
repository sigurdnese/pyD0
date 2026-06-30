import numpy as np
from iminuit import Minuit
from scipy.stats import chi2

# --- Model: Gaussian signal + pol2 background ---
def gaussian(x, mu, sigma, amp):
    norm = amp / (sigma * np.sqrt(2 * np.pi))
    return norm * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def background(x, c0, c1, c2):
    return c0 + c1 * x + c2 * x**2

def model_density(x, mu, sigma, amp, c0, c1, c2):
    # events / mass unit
    return gaussian(x, mu, sigma, amp) + background(x, c0, c1, c2)


def fit_mass_peak(
    h,
    axis_name="xaxis",
    x_range=None,
    mu_init=1.85,
    sigma_init=0.015,
    amp_frac_init=0.007,
    mu_limits=(1.8, 1.9),
    sigma_limits=(0.01, 0.025),
    amp_frac_limits=(0.001, 0.02),
    limit_tol=1e-6
):
    """
    Fit a 1D invariant-mass histogram with:
      Gaussian signal (total yield = amp)
      + quadratic background (c0 + c1*x + c2*x^2)
    using a binned Poisson likelihood.

    Parameters
    ----------
    h : hist.Hist
        1D histogram of invariant mass.
    axis_name : str
        Name of the axis in `h` containing the mass.
    mu_init, sigma_init : float
        Initial guesses for mean and width of the Gaussian.
    amp_frac_init : float
        Initial guess for signal yield as a fraction of total counts.
    mu_limits, sigma_limits : (float, float)
        Limits for mu and sigma.

    Returns
    -------
    result : dict
        {
          "params": dict of best-fit parameter values,
          "errors": dict of 1-sigma errors,
          "cov": 2D numpy array (covariance matrix),
          "chi2": Pearson chi^2,
          "ndof": degrees of freedom,
          "chi2_ndof": chi2 / ndof,
          "p_value": p-value for chi2,
          "signal_yield": best-fit signal yield (amp),
          "signal_yield_err": its uncertainty
        }
    """
    # --- Extract histogram info ---
    axis = h.axes[axis_name]
    bin_edges_full = axis.edges
    bin_centers_full = axis.centers
    counts_full = h.values()
    bin_widths_full = np.diff(bin_edges_full)
    
    # --- Restrict to x_range if requested ---
    if x_range is not None:
        xmin, xmax = x_range
        mask = (bin_centers_full >= xmin) & (bin_centers_full <= xmax)
        if not np.any(mask):
            raise ValueError("x_range excludes all bins; nothing to fit.")

        bin_centers = bin_centers_full[mask]
        counts = counts_full[mask]

        # For regular binning this is fine; for irregular, subset widths
        bin_edges = bin_edges_full
        bin_widths = bin_widths_full[mask]
    else:
        bin_edges = bin_edges_full
        bin_centers = bin_centers_full
        counts = counts_full
        bin_widths = bin_widths_full

    def expected_counts(mu, sigma, amp, c0, c1, c2):
        # expected events per bin
        return model_density(bin_centers, mu, sigma, amp, c0, c1, c2) * bin_widths

    # --- Poisson NLL ---
    def nll(mu, sigma, amp, c0, c1, c2):
        lam = expected_counts(mu, sigma, amp, c0, c1, c2)
        lam = np.clip(lam, 1e-12, None)
        return np.sum(lam - counts * np.log(lam))

    # --- Initial values ---
    total = counts.sum()
    amp0 = total * amp_frac_init
    c00 = total * (1.0 - amp_frac_init) / (bin_edges[-1] - bin_edges[0])

    m = Minuit(
        nll,
        mu=mu_init,
        sigma=sigma_init,
        amp=amp0,
        c0=c00,
        c1=0.0,
        c2=0.0,
    )

    # Limits
    m.limits["mu"] = mu_limits
    m.limits["sigma"] = sigma_limits
    m.limits["amp"] = (total * amp_frac_limits[0], total * amp_frac_limits[1])
    m.limits["c2"] = (None, None)

    # First pass
    m.migrad()

    max_passes = 3
    passes = 1
    # Do up to 2 extra passes if Minuit still says the minimum is not valid
    while (not m.valid) and (passes < max_passes):
        m.migrad()
        passes += 1

    if not m.valid:
        print("Warning: Migrad not valid *before* hesse call!")
        print(m.fmin)

    m.hesse()

    # --- Goodness of fit (Pearson chi^2) ---
    lam_best = expected_counts(
        m.values["mu"],
        m.values["sigma"],
        m.values["amp"],
        m.values["c0"],
        m.values["c1"],
        m.values["c2"],
    )
    mask = lam_best > 0
    chi2_val = np.sum((counts[mask] - lam_best[mask]) ** 2 / lam_best[mask])
    ndof = mask.sum() - m.nfit
    p_value = chi2.sf(chi2_val, ndof)

    # --- Signal yield ---
    signal_yield = m.values["amp"]
    signal_yield_err = m.errors["amp"]
    
    # --- Check if any parameter is at (or very close to) its limits ---
    at_limits = {}
    for name in m.parameters:
        lower, upper = m.limits[name]
        val = m.values[name]

        at_lower = False
        at_upper = False

        if lower is not None:
            # relative difference to lower
            scale = max(abs(lower), 1.0)
            if abs(val - lower) / scale < limit_tol:
                at_lower = True

        if upper is not None:
            # relative difference to upper
            scale = max(abs(upper), 1.0)
            if abs(val - upper) / scale < limit_tol:
                at_upper = True

        at_limits[name] = {
            "at_lower": at_lower,
            "at_upper": at_upper,
            "lower": lower,
            "upper": upper,
            "value": val,
        }
        
    # --- Package results ---
    params = {name: m.values[name] for name in m.parameters}
    errors = {name: m.errors[name] for name in m.parameters}

    result = {
        "params": params,
        "errors": errors,
        "chi2": chi2_val,
        "ndof": ndof,
        "chi2_ndof": chi2_val / ndof if ndof > 0 else np.nan,
        "p_value": p_value,
        "signal_yield": signal_yield,
        "signal_yield_err": signal_yield_err,
        "amp_frac": signal_yield / total,
        "at_limits": at_limits,     # per-parameter info about limits
        "minuit": m # Minuit object
    }

    return result
