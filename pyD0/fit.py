import numpy as np
from iminuit import Minuit
from scipy.stats import chi2
from scipy.stats import landau
from scipy.special import ndtr

# --- Model: Gaussian signal + pol2 background ---
def gaussian(x, mu, sigma, amp):
    norm = amp / (sigma * np.sqrt(2 * np.pi))
    return norm * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def background_quadratic(x, c0, c1, c2):
    return c0 + c1 * x + c2 * x**2

def model_density_quadradic(x, mu, sigma, amp, c0, c1, c2):
    # events / mass unit
    return gaussian(x, mu, sigma, amp) + background_quadratic(x, c0, c1, c2)

def background_exponential(x, b0, k):
    return b0*np.exp(k*x)

def model_density_exponential(x, mu, sigma, amp, b0, k):
    # events / mass unit
    return gaussian(x, mu, sigma, amp) + background_exponential(x, b0, k)

def fit_mass_peak(
    h,
    axis_name="xaxis",
    x_range=None,
    mu_init=1.85,
    sigma_init=0.015,
    amp_frac_init=0.007,
    mu_limits=(1.8, 1.9),
    sigma_limits=(0.01, 0.025),
    sigma_fix=None,
    amp_frac_limits=(0.001, 0.02),
    limit_tol=1e-6,
    background="quadratic",
    c2_limits=(None, None),
    exp_slope_init=0.0,
    exp_slope_limits=(None, None),
    exp_norm_limits=(0.0, None),
):
    """
    Fit a 1D invariant-mass histogram using a binned Poisson likelihood.

    Signal model
    ------------
    Gaussian with total yield ``amp``.

    Background models
    -----------------
    ``background="quadratic"``:
        c0 + c1*x + c2*x**2

    ``background="exponential"``:
        b0 * exp(k * x)

    Here ``b0`` is the background density at 0. The exponential slope ``k`` may be
    positive or negative.

    Parameters
    ----------
    h : hist.Hist
        One-dimensional invariant-mass histogram.
    axis_name : str
        Name of the mass axis.
    x_range : tuple[float, float] or None
        Optional fit range.
    background : {"quadratic", "exponential"}
        Background functional form.
    exp_slope_init : float
        Initial value of the exponential slope k.
    exp_slope_limits : tuple[float or None, float or None]
        Limits on the exponential slope k.
    exp_norm_limits : tuple[float or None, float or None]
        Limits on the exponential normalization b0.

    Returns
    -------
    dict
        Fit parameters, errors, covariance matrix, goodness-of-fit
        quantities, signal yield, and the underlying Minuit object.
    """

    if background not in {"quadratic", "exponential"}:
        raise ValueError(
            f"Unknown background={background!r}. "
            "Use 'quadratic' or 'exponential'."
        )

    # ------------------------------------------------------------------
    # Extract and select histogram bins
    # ------------------------------------------------------------------
    axis = h.axes[axis_name]
    edges_full = np.asarray(axis.edges)
    centers_full = np.asarray(axis.centers)
    counts_full = np.asarray(h.values())

    if x_range is not None:
        xmin, xmax = x_range
        fit_mask = (centers_full >= xmin) & (centers_full <= xmax)

        if not np.any(fit_mask):
            raise ValueError("x_range excludes all bins; nothing to fit.")
    else:
        fit_mask = np.ones_like(centers_full, dtype=bool)

    counts = counts_full[fit_mask]
    centers = centers_full[fit_mask]

    bin_lo = edges_full[:-1][fit_mask]
    bin_hi = edges_full[1:][fit_mask]
    bin_widths = bin_hi - bin_lo

    fit_xmin = bin_lo.min()
    fit_xmax = bin_hi.max()

    total = counts.sum()
    if total <= 0:
        raise ValueError("The selected fit range contains no entries.")

    # ------------------------------------------------------------------
    # Bin-integrated signal and background models
    # ------------------------------------------------------------------
    def gaussian_bin_yield(mu, sigma, amp):
        """
        Exact Gaussian yield in every bin.

        amp is the total Gaussian yield integrated over (-inf, +inf).
        """
        z_hi = (bin_hi - mu) / sigma
        z_lo = (bin_lo - mu) / sigma
        return amp * (ndtr(z_hi) - ndtr(z_lo))

    def quadratic_bin_yield(c0, c1, c2):
        """
        Integral of c0 + c1*x + c2*x^2 in each bin.
        """
        return (
            c0 * (bin_hi - bin_lo)
            + 0.5 * c1 * (bin_hi**2 - bin_lo**2)
            + (c2 / 3.0) * (bin_hi**3 - bin_lo**3)
        )

    def exponential_bin_yield(b0, k):
        """
        Integral of b0 * exp(k * x) in each bin.

        The k -> 0 limit is handled explicitly to avoid numerical issues.
        """
        if abs(k) < 1e-10:
            return b0 * bin_widths

        return (
            b0
            * (
                np.exp(k * (bin_hi))
                - np.exp(k * (bin_lo))
            )
            / k
        )

    # ------------------------------------------------------------------
    # Construct the likelihood for the selected background model
    # ------------------------------------------------------------------
    amp0 = total * amp_frac_init
    background_density0 = total * (1.0 - amp_frac_init) / (fit_xmax - fit_xmin)

    if background == "quadratic":

        def expected_counts(mu, sigma, amp, c0, c1, c2):
            return (
                gaussian_bin_yield(mu, sigma, amp)
                + quadratic_bin_yield(c0, c1, c2)
            )

        def nll(mu, sigma, amp, c0, c1, c2):
            lam = expected_counts(mu, sigma, amp, c0, c1, c2)
            lam = np.clip(lam, 1e-12, None)
            return np.sum(lam - counts * np.log(lam))

        m = Minuit(
            nll,
            mu=mu_init,
            sigma=sigma_init,
            amp=amp0,
            c0=background_density0,
            c1=0.0,
            c2=0.0,
        )

        m.limits["c2"] = c2_limits

    elif background == "exponential":

        def expected_counts(mu, sigma, amp, b0, k):
            return (
                gaussian_bin_yield(mu, sigma, amp)
                + exponential_bin_yield(b0, k)
            )

        def nll(mu, sigma, amp, b0, k):
            lam = expected_counts(mu, sigma, amp, b0, k)
            lam = np.clip(lam, 1e-12, None)
            return np.sum(lam - counts * np.log(lam))

        m = Minuit(
            nll,
            mu=mu_init,
            sigma=sigma_init,
            amp=amp0,
            b0=background_density0,
            k=exp_slope_init,
        )

        m.limits["b0"] = exp_norm_limits
        m.limits["k"] = exp_slope_limits

    else: 
        raise Exception(f"Background '{background}' not recognized!")

    # ------------------------------------------------------------------
    # Common parameter limits and minimization
    # ------------------------------------------------------------------
    m.errordef = Minuit.LIKELIHOOD

    m.limits["mu"] = mu_limits
    m.limits["sigma"] = sigma_limits
    if sigma_fix is not None:
        m.fixto("sigma", sigma_fix)

    if amp_frac_limits is not None:
        m.limits["amp"] = (
            total * amp_frac_limits[0],
            total * amp_frac_limits[1],
        )

    m.migrad()

    max_passes = 3
    passes = 1
    while not m.valid and passes < max_passes:
        m.migrad()
        passes += 1

    if not m.valid:
        print("Warning: Migrad was not valid before the Hesse call.")
        print(m.fmin)

    m.hesse()

    # ------------------------------------------------------------------
    # Goodness of fit
    # ------------------------------------------------------------------
    best_values = {name: m.values[name] for name in m.parameters}
    lam_best = expected_counts(**best_values)

    positive_mask = lam_best > 0
    chi2_val = np.sum(
        (counts[positive_mask] - lam_best[positive_mask]) ** 2
        / lam_best[positive_mask]
    )

    ndof = positive_mask.sum() - m.nfit
    p_value = chi2.sf(chi2_val, ndof) if ndof > 0 else np.nan

    # ------------------------------------------------------------------
    # Identify parameters close to their limits
    # ------------------------------------------------------------------
    at_limits = {}

    for name in m.parameters:
        lower, upper = m.limits[name]
        value = m.values[name]

        at_lower = False
        at_upper = False

        if lower is not None:
            scale = max(abs(lower), 1.0)
            at_lower = abs(value - lower) / scale < limit_tol

        if upper is not None:
            scale = max(abs(upper), 1.0)
            at_upper = abs(value - upper) / scale < limit_tol

        at_limits[name] = {
            "at_lower": at_lower,
            "at_upper": at_upper,
            "lower": lower,
            "upper": upper,
            "value": value,
        }

    params = {name: m.values[name] for name in m.parameters}
    errors = {name: m.errors[name] for name in m.parameters}

    return {
        "background": background,
        "params": params,
        "errors": errors,
        "cov": np.array(m.covariance) if m.covariance is not None else None,
        "chi2": chi2_val,
        "ndof": ndof,
        "chi2_ndof": chi2_val / ndof if ndof > 0 else np.nan,
        "p_value": p_value,
        "signal_yield": m.values["amp"],
        "signal_yield_err": m.errors["amp"],
        "amp_frac": m.values["amp"] / total,
        "at_limits": at_limits,
        "minuit": m,
    }

def model_exp(x, params):
    A = params[0]
    k = params[1]
    return A * np.exp(-k * x)

def model_linear(x, params):
    a = params[0]
    b = params[1]
    return a * x + b

def model_powerlaw(x, params):
    return params[0]*x/np.power((1+np.power(x/params[1],params[3])),params[2])

def model_landau(x, params):
    return params[0]*landau(x, params[1], params[2])

def model_pw_logparabola_exp(x, params):
    A = params[0]
    x0 = params[1]
    xp = params[2]
    T = params[3]
    return np.piecewise(x, [x <= x0, x > x0],
                        [lambda x: A*np.exp(((x0-xp)**2-(x-xp)**2)/(2*T*(x0-xp))),
                         lambda x: A*np.exp(-(x-x0)/T)])

def model_pw_logparabola_exp_nc(x, params):
    A = params[0]
    x0 = params[1]
    xp = params[2]
    T = params[3]
    sigma = params[4]
    return np.piecewise(x, [x <= x0, x > x0],
                        [lambda x: A*np.exp(((x0-xp)**2-(x-xp)**2)/(2*sigma**2)),
                         lambda x: A*np.exp(-(x-x0)/T)])

def model_diff_exp(x, params):
    A = params[0]
    T = params[1]
    q = params[2]
    k = params[3]
    return A*np.exp(-x/T)*(1-q*np.exp(-k*x))

def fit_ir_trend(
    h,
    initial_parameters,
    fixto=None,
    axis_name="xaxis",
    model=model_exp,
    limit_tol=1e-6,
    h_trials=None,
    h_successes=None,
    do_print=True
):
    """
    Fit data points along an interaction rate axis 
    using a binned Poisson likelihood.

    Parameters
    ----------
    h : hist.Hist
        1D histogram of data points.
    axis_name : str
        Name of the axis in `h` containing the IR.
    h_trials : hist.Hist
        The denominator when fitting efficiencies
    h_successes : hist.Hist
        The numerator when fitting efficiencies

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
        }
    """
    # --- Extract histogram info ---
    axis = h.axes[axis_name]
    bin_edges = axis.edges
    bin_centers = axis.centers
    counts = h.values()
    bin_widths = np.diff(bin_edges)
    
    def expected_counts(params):
        # expected events per bin
        return model(bin_centers, params)

    # --- Poisson NLL ---
    def nll(params):
        lam = expected_counts(params)
        lam = np.clip(lam, 1e-12, None)
        return np.sum(lam - counts * np.log(lam))

    # --- For efficiencies, use binomial NLL ---
    # Assume extact same binning for h_trials and h_successes as for h
    def nll_binomial(params):
        p = expected_counts(params)
        p = np.clip(p, 1e-12, None)
        return -np.sum(successes * np.log(p) + (trials - successes) * np.log(1 - p))

    if h_trials is None and h_successes is None:
        if do_print:
            print("Using Poisson NLL")
        m = Minuit(nll, initial_parameters)
    else: 
        if do_print:
            print("Using binomial NLL")
        trials = h_trials.values()
        successes = h_successes.values()
        m = Minuit(nll_binomial, initial_parameters)
    m.errordef = Minuit.LIKELIHOOD

    if fixto is not None:
        for i in range(len(initial_parameters)):
            if fixto[i] is not None:
                m.fixto(f"x{i}", fixto[i])

    # if model == model_exp:
    #     m.limits['x0'] = (0, None)
    #     m.limits['x1'] = (0, None)

    # First pass
    m.migrad()

    max_passes = 3
    passes = 1
    # Do up to 2 extra passes if Minuit still says the minimum is not valid
    while (not m.valid) and (passes < max_passes):
        m.migrad()
        passes += 1

    if not m.valid and do_print:
        print("Warning: Migrad not valid *before* hesse call!")
        print(m.fmin)

    # Compute uncertainties
    m.hesse()

    # --- Goodness of fit (Pearson chi^2) ---
    lam_best = expected_counts(m.values)
    mask = lam_best > 0
    chi2_val = np.sum((counts[mask] - lam_best[mask]) ** 2 / lam_best[mask])
    ndof = mask.sum() - m.nfit
    p_value = chi2.sf(chi2_val, ndof)

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
        "at_limits": at_limits,     # per-parameter info about limits
        "minuit": m # Minuit object
    }

    return result
