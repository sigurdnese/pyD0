import ROOT as r
from enum import Enum, auto
import hist
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
from deprecated import deprecated
import pandas as pd
from jacobi import propagate
import ctypes
from . import fit_functions
from . import utils
from . import fit

# Global list to keep callables alive
_tf1_callables = []
marker_styles = [20, 21, 22, 23, 33, 34, 47, 41]
default_palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:olive", "tab:cyan", "tab:pink"]

# Direct all prints in ROOT through its internal error handler (fix 'Q' option in TH1.Fit)
r.gPrintViaErrorHandler = True

DEFAULT_RUNLIST_PASS4 = ['545367', '545345', '545332', '545312', '545311', '545296', '545294', '545291', '545289', '545262', '545249', '545246', '545223', '545222', '545210', '545185', '545184', '545171', '545117', '545103', '545086', '545064', '545063', '545062', '545060', '545047', '545044', '545042', '545041', '545009', '545008', '545004', '544992', '544991', '544968', '544964', '544963', '544961', '544947', '544931', '544917', '544914', '544913', '544896', '544887', '544886', '544868', '544813', '544797', '544794', '544767', '544754', '544742', '544739', '544696', '544694', '544693', '544692', '544674', '544672', '544653', '544652', '544640', '544614', '544585', '544583', '544582', '544580', '544568', '544567', '544565', '544564', '544551', '544550', '544549', '544548', '544518', '544515', '544514', '544512', '544511', '544510', '544508', '544492', '544491', '544490', '544477', '544476', '544475', '544474', '544454', '544392', '544391', '544390', '544389', '544185', '544184', '544124', '544123', '544122', '544116', '544098', '544032', '544028', '544013']

RUNLIST_PASS4_UPCSETTINGS_YES = ['545246', '545311', '545262', '545222', '545367', '544754', '545117', '545332', '545041', '545171', '544886', '544794', '544913', '544914', '545042', '544961', '544692', '545062', '545008', '545210', '544742', '544887', '545009', '544454', '544511', '544098', '544568', '545044', '544475', '544964', '544694', '544390', '544512', '544868', '544583', '544476', '544122', '544550', '544696', '545249', '544551', '544391', '544514', '544917', '544492', '544968', '545063', '545047', '544585', '544515', '544477', '544392', '545064', '545185', '544797', '544992', '545312', '544896', '544518', '545223', '544124', '545291', '544013', '545294', '545296'] 

RUNLIST_PASS4_UPCSETTINGS_NO = ['544123', '544490', '544614', '544652', '544767', '544947', '544474', '544640', '544548', '544564', '544672', '544739', '544116', '544508', '544491', '544580', '544184', '544674', '544389', '544653', '544565', '544510', '544028', '544549', '544185', '544582', '545289', '545004', '544567', '545060', '544931', '545184']

DEFAULT_RUNLIST_PASS5 = ['545312', '545296', '545294', '545291', '545249', '545223', '545210', '545185', '545064', '545063', '545047', '545044', '545009', '544992', '544968', '544964', '544963', '544961', '544947', '544931', '544917', '544914', '544913', '544896', '544887', '544886', '544868', '544813', '544797', '544794', '544767', '544754', '544742', '544739', '544696', '544694', '544693', '544692', '544674', '544672', '544653', '544652', '544640', '544614', '544585', '544583', '544582', '544580', '544568', '544567', '544565', '544564', '544551', '544550', '544549', '544548', '544518', '544515', '544514', '544512', '544511', '544510', '544508', '544492', '544491', '544490', '544477', '544476', '544475', '544474', '544454', '544392', '544391', '544390', '544389', '544185', '544184', '544124', '544123', '544122', '544116', '544098', '544032', '544028', '544013', '545066', '545295']

RUNLIST_PASS5_CBT_HADRONPID = ['544614', '544640', '544674', '544754', '544767', '544868', '544886', '544887', '544896', '544913', '544914', '544917', '544931', '544947', '544961', '544963', '544964', '544968', '544992', '545009', '545044', '545047', '545063', '545064', '545066', '545185', '545210', '545223', '545249', '545291', '545294', '545295', '545296', '545312', '544813', '544028', '544032', '544013', '544794', '544797', '544098', '544116', '544122', '544123', '544124', '544184', '544185', '544389', '544390', '544391', '544392', '544454', '544474', '544475', '544476', '544477', '544490', '544491', '544492', '544508', '544510', '544511', '544548', '544549', '544550', '544551', '544564', '544565', '544567', '544568', '544580', '544582', '544583', '544585', '544652', '544653', '544692', '544693', '544694', '544696', '544739', '544742']

class FileStructures(Enum):
    RUNDIRS = auto()
    FLAT = auto()

def correct_efficiency_ir_dependence(hist_yield_per_lumi, hist_mc_rec_ir, hist_mc_gen_ir, savefig=None):
    """
    Correct the MC efficiency as a function of IR, using the data yield per lumi IR dependence.
    Exponential curves are fitted to the data yield per lumi and the MC efficiency.
    The data yield per lumi is normalized such that the fitted curves' values at IR=0 are equal.
    The curve R(IR) is constructed as the ratio between the MC fitted curve and the normalized data fitted curve.
    Then, the MC reconstructed counts histogram is rescaled by R(IR), obtaining a corrected MC efficiency in bins of IR.
    The corrected MC efficiency is then integrated over IR, using the MC generated counts as weights.
    The input histograms are integrated over y and pT, and the user must take care that the integration ranges correspond!
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 14
    })

    hist_mc_eff_ir = hist_mc_rec_ir.Clone()
    hist_mc_eff_ir.Divide(hist_mc_eff_ir, hist_mc_gen_ir, 1, 1, 'B')

    # Plot the MC efficiency and the yield per lumi
    fig1, ax1 = plt.subplots(1, 2, figsize=(20, 6))
    num_bins_mc = hist_mc_eff_ir.GetNbinsX()
    centers_mc = np.array([hist_mc_eff_ir.GetBinCenter(i) for i in range(1, num_bins_mc + 1)])
    counts_mc = np.array([hist_mc_eff_ir.GetBinContent(i) for i in range(1, num_bins_mc + 1)])
    widths_mc = np.array([hist_mc_eff_ir.GetBinWidth(i) for i in range(1, num_bins_mc + 1)])
    errors_mc = np.array([hist_mc_eff_ir.GetBinError(i) for i in range(1, num_bins_mc + 1)])
    ax1[0].errorbar(centers_mc, counts_mc, xerr=widths_mc/2, yerr=errors_mc, fmt='.', color='green', lw=1, label='MC efficiency', zorder=99)
    ax1[0].set_xlabel("Interaction rate (kHz)")
    ax1[0].legend(frameon=False)
    num_bins_data = hist_yield_per_lumi.GetNbinsX()
    centers_data = np.array([hist_yield_per_lumi.GetBinCenter(i) for i in range(1, num_bins_data + 1)])
    counts_data = np.array([hist_yield_per_lumi.GetBinContent(i) for i in range(1, num_bins_data + 1)])
    widths_data = np.array([hist_yield_per_lumi.GetBinWidth(i) for i in range(1, num_bins_data + 1)])
    errors_data = np.array([hist_yield_per_lumi.GetBinError(i) for i in range(1, num_bins_data + 1)])
    ax1[1].errorbar(centers_data, counts_data, xerr=widths_data/2, yerr=errors_data, fmt='.', color='blue', lw=1, label=f'Data yield per lumi', zorder=99)
    ax1[1].set_xlabel("Interaction rate (kHz)")

    # Convert histograms to scikit-hep/hist and fit them using iminuit
    h_yield_per_lumi = utils.root_to_hist(hist_yield_per_lumi)
    print("----- Fitting the data yield per lumi -----")
    fit_data = fit.fit_ir_trend(h_yield_per_lumi, [0, 0], axis_name=h_yield_per_lumi.axes[0].name, model=fit.model_exp)
    display(fit_data['minuit'])
    print("")

    h_eff_ir = utils.root_to_hist(hist_mc_eff_ir)
    h_mc_rec_ir = utils.root_to_hist(hist_mc_rec_ir)
    h_mc_gen_ir = utils.root_to_hist(hist_mc_gen_ir)
    # Remove trailing zeros from the MC histograms
    (h_mc_gen_ir, h_eff_ir, h_mc_rec_ir), trim = utils.trim_trailing_zeros(h_mc_gen_ir, h_eff_ir, h_mc_rec_ir)
    # Fit the MC efficiency, supplying the rec and gen histograms so that the binomial NLL can be used
    print("----- Fitting the MC efficiency -----")
    fit_mc = fit.fit_ir_trend(h_eff_ir, [0, 0], axis_name=h_eff_ir.axes[0].name, model=fit.model_exp, h_successes=h_mc_rec_ir, h_trials=h_mc_gen_ir)
    display(fit_mc['minuit'])
    print("")

    # Get the error bands for the fits
    params_data = fit_data['minuit'].values
    params_mc = fit_mc['minuit'].values
    cov_data = fit_data['minuit'].covariance
    cov_mc = fit_mc['minuit'].covariance
    ir_axis = np.linspace(0, 50, 500)
    y_mc_fit, ycov_mc_fit = propagate(lambda p: fit.model_exp(ir_axis, p), params_mc, cov_mc)
    y_mc_fit_errprop = np.diag(ycov_mc_fit) ** 0.5
    ax1[0].plot(ir_axis, y_mc_fit, color='C3', label=f"Fit: {params_mc[0]:.5g}*exp(-{params_mc[1]:.5g}*IR)")
    ax1[0].fill_between(ir_axis, y_mc_fit-y_mc_fit_errprop, y_mc_fit+y_mc_fit_errprop, facecolor='C3', alpha=0.5)
    y_data_fit, ycov_data_fit = propagate(lambda p: fit.model_exp(ir_axis, p), params_data, cov_data)
    y_data_fit_errprop = np.diag(ycov_data_fit) ** 0.5
    ax1[1].plot(ir_axis, y_data_fit, color='C1', label=f"Fit: {params_data[0]:.5g}*exp(-{params_data[1]:.5g}*IR)")
    ax1[1].fill_between(ir_axis, y_data_fit-y_data_fit_errprop, y_data_fit+y_data_fit_errprop, facecolor='C1', alpha=0.5)

    # Show the data points and fitted curves with error bands
    ax1[0].legend(frameon=False)
    ax1[1].legend(frameon=False)
    plt.show()

    # Combined parameter vector and covariance
    n_data = len(params_data)
    n_mc = len(params_mc)
    theta = np.concatenate([params_data, params_mc])
    # Assume no cross-covariance
    cov_theta = np.block([
        [cov_data,            np.zeros((n_data, n_mc))],
        [np.zeros((n_mc, n_data)), cov_mc             ]
    ])

    ir0 = 0.0
    # Extrapolated values at ir0
    f_data_ir0 = fit.model_exp(ir0, params_data)
    f_mc_ir0 = fit.model_exp(ir0, params_mc)
    # Normalization factor
    s = f_mc_ir0 / f_data_ir0
    # Define the function which normalizes the data, so that we can propagate the error for the normalized curve
    def model_exp_norm(ir, theta):
        params_data_curr = theta[:n_data]
        params_mc_curr = theta[n_data:]
        # Extrapolated values at ir0
        f_data_ir0_curr = fit.model_exp(ir0, params_data_curr)
        f_mc_ir0_curr = fit.model_exp(ir0, params_mc_curr)
        # Normalization factor
        s_curr = f_mc_ir0_curr / f_data_ir0_curr
        return s_curr * fit.model_exp(ir, params_data_curr)
    # Obtain the normalized data curve and its error band
    y_data_norm, ycov_data_norm = propagate(lambda p: model_exp_norm(ir_axis, p), theta, cov_theta)
    y_data_norm_errprop = np.diag(ycov_data_norm) ** 0.5

    # Plot the MC points and its fit, and the normalized data points and the normalized fit in the same figure
    fig2 = plt.figure(figsize=(8,8))
    gs = fig2.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.)
    ax_top = fig2.add_subplot(gs[0, 0])
    ax_bottom = fig2.add_subplot(gs[1, 0], sharex=ax_top)

    widths_mc = h_eff_ir.axes[0].edges[1:] - h_eff_ir.axes[0].edges[:-1]
    ax_top.errorbar(h_eff_ir.axes[0].centers, h_eff_ir.values(), xerr=widths_mc/2, yerr=h_eff_ir.variances()**0.5, fmt='.', color='green', lw=1, label=f'MC efficiency', zorder=0)
    ax_top.errorbar(centers_data, counts_data * s, xerr=widths_data/2, yerr=errors_data * s, fmt='.', color='blue', lw=1, label=f'Data yield per lumi normalized to MC at IR=0', zorder=99)

    ax_top.plot(ir_axis, y_mc_fit, color='C3')#, label=f'Fit to exponential function')
    ax_top.fill_between(ir_axis, y_mc_fit-y_mc_fit_errprop, y_mc_fit+y_mc_fit_errprop, facecolor='C3', alpha=0.5)

    ax_top.plot(ir_axis, y_data_norm, color='C1')#, label=f'Data curve normalized to MC at IR=0')
    ax_top.fill_between(ir_axis, y_data_norm-y_data_norm_errprop, y_data_norm+y_data_norm_errprop, facecolor='C1', alpha=0.5)

    ax_top.legend(frameon=False)

    # Use the ratio between the data curve and the MC curve to rescale the reconstructed counts
    # Define the ratio itself as a function, to make it available for error propagation
    def ratio_data_norm_mc_fit(ir, theta):
        params_data_curr = theta[:n_data]
        params_mc_curr = theta[n_data:]
        # Extrapolated values at ir0
        f_data_ir0_curr = fit.model_exp(ir0, params_data_curr)
        f_mc_ir0_curr = fit.model_exp(ir0, params_mc_curr)
        # Normalization factor
        s_curr = f_mc_ir0_curr / f_data_ir0_curr
        return s_curr * fit.model_exp(ir, params_data_curr) / fit.model_exp(ir, params_mc_curr)

    # Calculate the ratio and its error
    R_ir, R_ir_cov = propagate(lambda p: ratio_data_norm_mc_fit(ir_axis, p), theta, cov_theta)
    R_ir_errprop = np.diag(R_ir_cov) ** 0.5
    ax_bottom.plot(ir_axis, R_ir, color='C4', label=f'Correction factor for MC efficiency')
    ax_bottom.fill_between(ir_axis, R_ir-R_ir_errprop, R_ir+R_ir_errprop, facecolor='C4', alpha=0.5)
    ax_bottom.set_xlabel("Interaction rate (kHz)")
    ax_bottom.legend(frameon=False)
    ax_bottom.set_xlabel("Interaction rate (kHz)")

    if savefig is not None:
        fig2.savefig(savefig)
    plt.show()

    # Perform the rescaling on the ROOT histograms
    hist_mc_rec_ir_rescaled = hist_mc_rec_ir.Clone()
    for i in range(hist_mc_rec_ir_rescaled.GetNbinsX()):
        ir_i = hist_mc_rec_ir_rescaled.GetBinCenter(i+1)
        N_rec_i = hist_mc_rec_ir.GetBinContent(i+1)
        err_rec_i = hist_mc_rec_ir.GetBinError(i+1)
        rescaling_factor_i, rescaling_factor_variance_i = propagate(lambda p: ratio_data_norm_mc_fit(ir_i, p), theta, cov_theta)
        hist_mc_rec_ir_rescaled.SetBinContent(i+1, N_rec_i * rescaling_factor_i) 
        hist_mc_rec_ir_rescaled.SetBinError(i+1, np.sqrt(N_rec_i**2 * rescaling_factor_variance_i + rescaling_factor_i**2 * err_rec_i**2))

    # Draw the original and rescaled rec. MC histogram
    c1 = r.TCanvas('c1', 'c1', 1200, 400)
    c1.Divide(2,1)
    c1.cd(1)
    hist_mc_rec_ir.Draw()
    hist_mc_rec_ir.SetTitle("Reconstructed D0 counts")
    hist_mc_rec_ir.SetStats(0)
    hist_mc_rec_ir_rescaled.Draw('same')
    hist_mc_rec_ir_rescaled.SetLineColor(r.kRed)
    l11 = r.TLegend(0.7, 0.75, 0.85, 0.85)
    l11.SetBorderSize(0)
    l11.AddEntry(hist_mc_rec_ir, "Original")
    l11.AddEntry(hist_mc_rec_ir_rescaled, "Rescaled")
    l11.Draw()
    # Draw the original and rescaled efficiency histograms
    c1.cd(2)
    hist_mc_eff_ir_rescaled = hist_mc_rec_ir_rescaled.Clone()
    hist_mc_eff_ir_rescaled.SetName('hist_mc_eff_ir_rescaled')
    hist_mc_eff_ir_rescaled.Divide(hist_mc_rec_ir_rescaled, hist_mc_gen_ir, 1, 1, 'B')
    hist_mc_eff_ir_rescaled.SetTitle("MC efficiency")
    hist_mc_eff_ir_rescaled.Draw()
    hist_mc_eff_ir_rescaled.SetLineColor(r.kRed)
    hist_mc_eff_ir.Draw('same')
    l12 = r.TLegend(0.7, 0.75, 0.85, 0.85)
    l12.SetBorderSize(0)
    l12.AddEntry(hist_mc_eff_ir, "Original")
    l12.AddEntry(hist_mc_eff_ir_rescaled, "Rescaled")
    l12.Draw()

    _tf1_callables.append([c1, hist_mc_rec_ir, hist_mc_rec_ir_rescaled, l11,
                           hist_mc_eff_ir, hist_mc_eff_ir_rescaled, l12])
    c1.Draw()

    # Now compute the efficiency integrated over IR in one go, to make error propagation simpler
    eff_rescaled_integrated, eff_rescaled_integrated_err_total, eff_rescaled_integrated_err_stat, eff_rescaled_integrated_err_fit = apply_ir_rescaling(hist_mc_rec_ir, hist_mc_gen_ir, theta, cov_theta, ir0, trim, n_data)

    # For comparison, calculate the original integrated efficiency and report the change in efficiency and cross section caused by the rescaling
    gen_orig_integrated_err = ctypes.c_double(0.0)
    rec_orig_integrated_err = ctypes.c_double(0.0)
    gen_orig_integrated = hist_mc_gen_ir.IntegralAndError(0, -1, err=gen_orig_integrated_err)
    rec_orig_integrated = hist_mc_rec_ir.IntegralAndError(0, -1, err=rec_orig_integrated_err)
    eff_orig_integrated = rec_orig_integrated / gen_orig_integrated
    eff_orig_integrated_error = np.sqrt((eff_orig_integrated * (1 - eff_orig_integrated)) / gen_orig_integrated)
    relative_change_eff = (eff_rescaled_integrated - eff_orig_integrated) / (eff_orig_integrated)
    relative_change_cs = (1/eff_rescaled_integrated - 1/eff_orig_integrated) / (1/eff_orig_integrated)

    print("----- Results for the rescaled efficiency integrated over IR -----")
    print(f"Error on integrated efficiency from rescaling procedure: {eff_rescaled_integrated_err_fit*100:.5f}")
    print(f"Error on integrated efficiency from statistical errors on MC counts: {eff_rescaled_integrated_err_stat*100:.5f}")
    print(f"Result for rescaled, IR integrated efficiency, with total error:") 
    print(f"    {eff_rescaled_integrated*100:.3f}% +- {eff_rescaled_integrated_err_total*100:.3f}%")
    print(f"Original integrated efficiency: {eff_orig_integrated*100:.3f} +- {eff_orig_integrated_error*100:.5f}%")
    print(f"Rescaling caused {relative_change_eff*100:.1f}% change in efficiency -> {relative_change_cs*100:.1f}% change in cross section")
    print("------------------------------------------------------------------")

    # Return the results, and all that is needed to apply the rescaling to a given efficiency, with error propagation
    result = {
        "ir_axis": ir_axis,
        "R_ir": R_ir,
        "theta": theta,
        "cov_theta": cov_theta,
        "eff_rescaled": eff_rescaled_integrated,
        "eff_rescaled_err": eff_rescaled_integrated_err_total
    }
    return result

def apply_ir_rescaling(hist_mc_rec_ir, hist_mc_gen_ir, theta, cov_theta, ir0, trim, n_data):
    # Obtain ir bins, MC rec and gen counts as arrays
    n_bins = hist_mc_rec_ir.GetNbinsX()
    ir_bin_centers_arr = np.array([hist_mc_rec_ir.GetBinCenter(i) for i in range(1, n_bins+1-trim)], dtype=float)
    N_gen_mc_arr = np.array([hist_mc_gen_ir.GetBinContent(i) for i in range(1, n_bins+1-trim)], dtype=float)
    N_rec_mc_arr = np.array([hist_mc_rec_ir.GetBinContent(i) for i in range(1, n_bins+1-trim)], dtype=float)
    # Raw MC efficiencies, bin by bin. Treated as deterministic wrt the fit parameter error propagation
    eff_mc_arr = N_rec_mc_arr / N_gen_mc_arr
    # Define the function which computes the rescaled efficiency
    def eff_rescaled(theta):
        params_data_curr = theta[:n_data]
        params_mc_curr = theta[n_data:]
        
        # Compute the normalization factor for the data curve
        eff_data_0 = fit.model_exp(ir0, params_data_curr)
        eff_mc_0 = fit.model_exp(ir0, params_mc_curr)
        s_curr = eff_mc_0 / eff_data_0
        
        num = 0.0
        den = 0.0
        for ir_i, Ngen_i, eff_mc_bin_i in zip(ir_bin_centers_arr, N_gen_mc_arr, eff_mc_arr):
            # Curve values at IR_i
            eff_data_fit_i = fit.model_exp(ir_i, params_data_curr)
            eff_mc_fit_i = fit.model_exp(ir_i, params_mc_curr)
            
            # Normalized data efficiency at IR_i
            eff_data_fit_norm_i = s_curr * eff_data_fit_i
            
            # Correction factor R(IR_i)
            R_i = eff_data_fit_norm_i / eff_mc_fit_i

            # Corrected efficiency in bin i
            eff_corrected_i = R_i * eff_mc_bin_i

            num += Ngen_i * eff_corrected_i
            den += Ngen_i

        return num / den

    eff_rescaled_integrated, eff_rescaled_integrated_var_fit = propagate(eff_rescaled, theta, cov_theta)
    eff_rescaled_integrated_err_fit = np.sqrt(eff_rescaled_integrated_var_fit)

    # Compute the binomial error due to the original statistical error bars on the MC histograms
    N_gen_mc_arr = np.array([hist_mc_gen_ir.GetBinContent(i) for i in range(1, n_bins+1-trim)], dtype=float)
    N_rec_mc_arr = np.array([hist_mc_rec_ir.GetBinContent(i) for i in range(1, n_bins+1-trim)], dtype=float)
    # Raw MC efficiencies, bin by bin. Treated as deterministic wrt the fit parameter error propagation
    eff_mc_arr = N_rec_mc_arr / N_gen_mc_arr
    eff_var_stat_arr = eff_mc_arr * (1.0 - eff_mc_arr) / N_gen_mc_arr
    # Propagate the statistical errors through the rescaling using the central parameters
    f_data_ir0 = fit.model_exp(ir0, theta[:n_data])
    f_mc_ir0 = fit.model_exp(ir0, theta[n_data:])
    s = f_mc_ir0 / f_data_ir0
    R_arr = np.array([
        (s * fit.model_exp(ir_i, theta[:n_data])) / fit.model_exp(ir_i, theta[n_data:])
        for ir_i in ir_bin_centers_arr
    ])
    eff_rescaled_var_stat_arr = (R_arr**2) * eff_var_stat_arr
    # Propagate the statistical errors through the integration over IR
    eff_rescaled_integrated_var_stat = np.sum((N_gen_mc_arr**2) * eff_rescaled_var_stat_arr) / (np.sum(N_gen_mc_arr)**2)
    eff_rescaled_integrated_err_stat = np.sqrt(eff_rescaled_integrated_var_stat)
    # Combine the stat error and the error from the rescaling procedure in quadrature
    eff_rescaled_integrated_err_total = np.sqrt(eff_rescaled_integrated_var_stat + eff_rescaled_integrated_var_fit)

    # Return the rescaled, integrated efficiency and the errors
    return eff_rescaled_integrated, eff_rescaled_integrated_err_total, eff_rescaled_integrated_err_stat, eff_rescaled_integrated_err_fit

def normalized_yield_ir(runlist, n_ir_bins=None, hist_dict=None, data_dir=None, file_structure=FileStructures.RUNDIRS, ncols=4, bin_edges_ir=None, norm_by='lumi', min_ir=0, max_ir=None):
    """
    Obtain D0 yield per ZNC luminosity or number of selected events as a function of ZNC hadronic interaction rate
    IR bins are chosen by equally distributing the D0 candidates among n_ir_bins bins along the IR axis
    Yield is extracted using iminuit 
    """
    if hist_dict is None:
        hist_names = ['analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz/MyMassInteractionRateHisto']
        if norm_by == 'selected_events':
            hist_names.append('analysis-event-selection/output;1/Event_AfterCuts/MyInteractionRateHisto') 
        hist_dict = get_histograms(data_dir, runlist, hist_names, histogramNamesfileStructure=file_structure)
    hist_m_ir = hist_dict['analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz/MyMassInteractionRateHisto']
    # Get IR bin edges
    if bin_edges_ir is None:
        min_bin = hist_m_ir.GetYaxis().FindBin(min_ir)
        max_bin = hist_m_ir.GetYaxis().FindBin(max_ir) - 1
        print(f"Finding IR bin edges based on equal division of D0 candidates between bin {min_bin} and {max_bin}")
        bin_edges_ir = utils.equal_stat_y_slices(hist_m_ir, n_ir_bins, min_bin, max_bin)
    for i, (low, high) in enumerate(bin_edges_ir):
        print(f"Range {i+1}: bin {low} -> {high}; IR={hist_m_ir.GetYaxis().GetBinLowEdge(low):.2f} kHz -> {hist_m_ir.GetYaxis().GetBinLowEdge(high):.2f} kHz")
        integral = hist_m_ir.Integral(binx1=1, binx2=hist_m_ir.GetXaxis().GetNbins(), biny1=low, biny2=high)
        print(f"  Integral={integral:.2f}")
    # Get slices of invariant mass
    m_projs = []
    for i, (ybin_lo, ybin_hi) in enumerate(bin_edges_ir):
        proj_name = f"mproj_slice{i}"
        proj = hist_m_ir.ProjectionX(proj_name, ybin_lo, ybin_hi)  # inclusive
        m_projs.append(proj)
        m_projs[-1].SetTitle(f"{hist_m_ir.GetYaxis().GetBinLowEdge(ybin_lo):.1f} #leq IR #leq {hist_m_ir.GetYaxis().GetBinUpEdge(ybin_hi):.1f} kHz")
    # Convert pyroot histograms to hist histograms
    m_projs_h = [utils.root_to_hist(h) for h in m_projs]
    for h in m_projs_h:
        h.axes[0].label = h.axes[0].label.replace("#leq", "$\\leq$")

    # Extract the yield and plot
    fit_results = []

    nrows = int(np.ceil(len(m_projs_h)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(23, 4*nrows))
    for i, h in enumerate(m_projs_h):
        row = int(np.floor(i/ncols))
        col = i%ncols
        ax=axes[row,col]
        
        axis = h.axes[0]
        bin_edges = axis.edges
        bin_centers = axis.centers
        counts = h.values()
        bin_widths = np.diff(bin_edges)

        fit_result = fit.fit_mass_peak(h, axis_name=axis.name, x_range=None, sigma_limits=(0.005, 0.05), amp_frac_init=0.007)
        fit_results.append(fit_result)
        
        # Plot data + fit
        x_plot = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        
        # density (events / GeV)
        y_density_total = fit.model_density(x_plot, *fit_result["params"].values())
        y_density_sig   = fit.gaussian(x_plot, fit_result["params"]["mu"], fit_result["params"]["sigma"], fit_result["params"]["amp"])
        y_density_bkg   = fit.background(x_plot, fit_result["params"]["c0"], fit_result["params"]["c1"], fit_result["params"]["c2"])
        
        # convert to "events per bin" for plotting on histogram scale
        bin_width_plot = bin_widths[0]  # regular binning; for irregular, scale differently
        y_plot_total = y_density_total * bin_width_plot
        y_plot_sig   = y_density_sig * bin_width_plot
        y_plot_bkg   = y_density_bkg * bin_width_plot
        
        ax.errorbar(
            bin_centers,
            counts,
            yerr=np.sqrt(counts),
            fmt=".",
            color="black",
            label="Data",
        )
        
        ax.plot(x_plot, y_plot_total, color="red",  label="Signal + background")
        #ax.plot(x_plot, y_plot_sig,   color="blue", linestyle="--", label="Signal")
        ax.plot(x_plot, y_plot_bkg,   color="green", linestyle=":", label="Background")
            
        text = f"Yield={fit_result['signal_yield']:.0f} $\\pm$ {fit_result['signal_yield_err']:.0f}"
        text += f"\n $\\sigma$={fit_result['params']['sigma']:.4f}"
        text += f"\n $\\chi^{2}$/ndof = {fit_result['chi2_ndof']:.3f}"
        text += f"\n amp frac = {fit_result['amp_frac']:.4f}"
        text += f"\n valid = {fit_result['minuit'].valid}"
        any_at_limits = False
        for par, info in fit_result["at_limits"].items():
            if info["at_lower"] or info["at_upper"]:
                any_at_limits = True
                text += f"\n{par} is at a limit: \n   lower={info['at_lower']}\n   upper={info['at_upper']}"
        ax.text(0.7, 0.99, text, ha='left', va='top', transform=ax.transAxes)
        
        ax.set_xlabel("m [GeV]")
        ax.set_ylabel("Events / bin")
        ax.legend()
        ax.set_title(f"{h.axes[0].label}")

    plt.tight_layout()
    plt.show()

    # Get the IR values at the bin edges
    low_edges_ir = [hist_m_ir.GetYaxis().GetBinLowEdge(p[0]) for p in bin_edges_ir]
    low_edges_ir.append(hist_m_ir.GetYaxis().GetBinUpEdge(bin_edges_ir[-1][1]))
    
    if norm_by == 'selected_events':
        result_name = 'hist_yield_per_selected_event'
        result_title = "D0 yield / Selected events"
        denominator_name = 'hist_events_ir'
        hist_events_ir_fine = hist_dict['analysis-event-selection/output;1/Event_AfterCuts/MyInteractionRateHisto']
        hist_denominator = hist_events_ir_fine.Rebin(len(low_edges_ir) - 1, denominator_name, np.asarray(low_edges_ir, 'd'))
    elif norm_by == 'lumi':
        result_name = 'hist_yield_per_lumi'
        result_title = "D0 yield / ZNC lumi (µb)"
        denominator_name = 'hist_lumi_ir'
        # Integrate lumi inside the IR bins, using the instantaneous IR as a function of time
        hist_denominator = utils.get_lumi_vs_ir(runlist, np.array(low_edges_ir), do_print=False)
        for i in range(hist_denominator.GetNbinsX()):
            hist_denominator.SetBinError(i+1, hist_denominator.GetBinContent(i+1) * (5/219.1)) # Relative uncertainty of ZNC cross section, from https://alice-notes.web.cern.ch/system/files/notes/analysis/1515/2024-10-23-lumi_2024_v5_0.pdf 

    # Constuct raw yield vs IR histogram
    hist_yield = hist_denominator.Clone()
    hist_yield.Reset()
    hist_yield.SetTitle("")
    hist_yield.GetYaxis().SetTitle("Raw D0 yield")
    for i in range(hist_yield.GetNbinsX()):
        hist_yield.SetBinContent(i+1, fit_results[i]["signal_yield"])
        hist_yield.SetBinError(i+1, fit_results[i]["signal_yield_err"])

    # Constuct yield per lumi histogram
    hist_yield_normalized = hist_yield.Clone()
    hist_yield_normalized.SetName(result_name)
    hist_yield_normalized.GetYaxis().SetTitle(result_title)
    hist_yield_normalized.Divide(hist_denominator)

    # Draw the final result
    c = r.TCanvas()
    c.cd()
    hist_yield_normalized.Draw()
    hist_yield_normalized.GetXaxis().SetRangeUser(0, hist_yield_normalized.GetXaxis().GetBinUpEdge(hist_yield_normalized.GetNbinsX()))
    c.Draw()

    result = {
        'bin_edges_ir': bin_edges_ir,
        'low_edges_ir': low_edges_ir,
        result_name: hist_yield_normalized,
        'hist_yield': hist_yield,
        denominator_name: hist_denominator
    }

    return result

def selected_events_per_lumi_ir(runlist, n_ir_bins=None, hist_dict=None, data_dir=None, file_structure=FileStructures.RUNDIRS, ncols=4, bin_edges_ir=None, min_ir=0, max_ir=None):
    """
    Obtain D0 yield per ZNC luminosity or number of selected events as a function of ZNC hadronic interaction rate
    IR bins are chosen by equally distributing the D0 candidates among n_ir_bins bins along the IR axis
    Yield is extracted using iminuit 
    """
    if hist_dict is None:
        hist_names = ['analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz/MyMassInteractionRateHisto',
                      'analysis-event-selection/output;1/Event_AfterCuts/MyInteractionRateHisto']
        hist_dict = get_histograms(data_dir, runlist, hist_names, histogramNamesfileStructure=file_structure)
    hist_m_ir = hist_dict['analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz/MyMassInteractionRateHisto']
    # Even if we don't use the D0 yield, base the IR bins on the D0 candidate statistics
    if bin_edges_ir is None:
        min_bin = hist_m_ir.GetYaxis().FindBin(min_ir)
        max_bin = hist_m_ir.GetYaxis().FindBin(max_ir) - 1
        print(f"Finding IR bin edges based on equal division of D0 candidates between bin {min_bin} and {max_bin}")
        bin_edges_ir = utils.equal_stat_y_slices(hist_m_ir, n_ir_bins, min_bin, max_bin)
    for i, (low, high) in enumerate(bin_edges_ir):
        print(f"Range {i+1}: bin {low} -> {high}; IR={hist_m_ir.GetYaxis().GetBinLowEdge(low):.2f} kHz -> {hist_m_ir.GetYaxis().GetBinLowEdge(high):.2f} kHz")
        integral = hist_m_ir.Integral(binx1=1, binx2=hist_m_ir.GetXaxis().GetNbins(), biny1=low, biny2=high)
        print(f"  Integral={integral:.2f}")

    # Get the IR values at the bin edges
    low_edges_ir = [hist_m_ir.GetYaxis().GetBinLowEdge(p[0]) for p in bin_edges_ir]
    low_edges_ir.append(hist_m_ir.GetYaxis().GetBinUpEdge(bin_edges_ir[-1][1]))
    
    hist_events_ir_fine = hist_dict['analysis-event-selection/output;1/Event_AfterCuts/MyInteractionRateHisto']
    hist_events_ir = hist_events_ir_fine.Rebin(len(low_edges_ir) - 1, 'hist_events_ir', np.asarray(low_edges_ir, 'd'))

    # Integrate lumi inside the IR bins, using the instantaneous IR as a function of time
    hist_lumi_ir = utils.get_lumi_vs_ir(runlist, np.array(low_edges_ir), do_print=False)
    for i in range(hist_lumi_ir.GetNbinsX()):
        hist_lumi_ir.SetBinError(i+1, hist_lumi_ir.GetBinContent(i+1) * (5/219.1)) # Relative uncertainty of ZNC cross section, from https://alice-notes.web.cern.ch/system/files/notes/analysis/1515/2024-10-23-lumi_2024_v5_0.pdf 

    # Constuct events per lumi histogram
    hist_events_per_lumi = hist_events_ir.Clone()
    hist_events_per_lumi.SetName('hist_events_per_lumi')
    hist_events_per_lumi.SetTitle('Number of selected events per ZNC lumi')
    hist_events_per_lumi.GetYaxis().SetTitle("Selected events / ZNC lumi (µb)")
    hist_events_per_lumi.Divide(hist_lumi_ir)

    # Draw the final result
    print("Drawing...")
    c = r.TCanvas()
    c.cd()
    hist_events_per_lumi.Draw()
    hist_events_per_lumi.GetXaxis().SetRangeUser(0, hist_events_per_lumi.GetXaxis().GetBinUpEdge(hist_events_per_lumi.GetNbinsX()))
    c.Draw()

    result = {
        'bin_edges_ir': bin_edges_ir,
        'low_edges_ir': low_edges_ir,
        'hist_events_ir': hist_events_ir,
        'hist_lumi_ir': hist_lumi_ir,
        'hist_events_per_lumi': hist_events_per_lumi,
        'canvas': c
    }

    return result

def project_fiducial_acceptance(hist, minY, maxY):
    # Project a pT histogram from a y vs pT histogram
    lowerYBin = hist.GetXaxis().FindBin(minY)
    upperYBin = hist.GetXaxis().FindBin(maxY) - 1
    resultHist = hist.ProjectionY(f"{hist.GetName()}_y_{minY}_{maxY}", lowerYBin, upperYBin)
    return resultHist

def get_histograms(directory, runList, fullHistogramNames, histogramNamesfileStructure = FileStructures.FLAT, fileName = 'AnalysisResults.root'):
    splitHistNames = [s.split("/") for s in fullHistogramNames] 
    namesDict = {}
    for irow, (row, fullName) in enumerate(zip(splitHistNames, fullHistogramNames)):
        depth = len(row)
        if row[1] == 'output;1' or row[1] == 'Statistics;1':
            depth -= 1
            row[0] += ('/' + row[1])
            row = np.delete(row, 1)
        elif row[0] == 'eventselection-run3' and 'luminosity' in row[1]:
            # Special case: lumi hist can be accessed directly
            depth = 1
            namesDict[fullName] = fullName
        if depth == 3:
            if row[0] not in namesDict:
                namesDict[row[0]] = {row[1] : []}
            elif row[1] not in namesDict[row[0]]:
                namesDict[row[0]][row[1]] = []
            namesDict[row[0]][row[1]].append(row[2])
        elif depth == 2: # e. g. the lumi histogram
            if row[0] not in namesDict:
                namesDict[row[0]] = []
            namesDict[row[0]].append(row[1])
        elif depth == 1:
            pass
        else:
            raise Exception(f"Histogram name with depth = {depth}, don't know what to do!")
    histograms = {}
    print(f"Getting histograms from {directory}...")
    if depth == 3:
        for irun, run in enumerate(runList):
            print(f"Processing run {run} ({irun+1}/{len(runList)})...            ", end='\r')
            if (histogramNamesfileStructure == FileStructures.FLAT):
                filePath = f"{directory}/AnalysisResults_run{run}.root"
            elif (histogramNamesfileStructure == FileStructures.RUNDIRS):
                filePath = f"{directory}/{run}/{fileName}"
            else:
                raise Exception("Not a valid FileStructure!")
            with uproot.open(filePath) as file:
                for dirName in namesDict.keys():
                    try:
                        dir = file[dirName]
                    except:
                        raise Exception(f"Couldn't get '{dirName}' from file '{filePath}'!")
                    for i, item in enumerate(dir):
                        if item.member("fName") in namesDict[dirName]:
                            for ii, iitem in enumerate(dir[i]):
                                if iitem.member("fName") in namesDict[dirName][item.member("fName")]:
                                    fullName = dirName + '/' + item.member("fName") + '/' + iitem.member("fName")
                                    tmpHist = dir[i][ii]
                                    tmpHistWritable = tmpHist.to_writable()
                                    tmpHistPr = tmpHistWritable.to_pyroot()
                                    if tmpHistPr.GetEntries() == 0:
                                        print(f"WARNING: Histogram '{fullName}' in '{filePath}' has no entries!")
                                    if fullName not in histograms:
                                        histograms[fullName] = tmpHistPr
                                    else:
                                        histograms[fullName].Add(tmpHistPr)
    elif depth == 2:
        for irun, run in enumerate(runList):
            print(f"Processing run {run} ({irun+1}/{len(runList)})...            ", end='\r')
            if (histogramNamesfileStructure == FileStructures.FLAT):
                filePath = f"{directory}/AnalysisResults_run{run}.root"
            elif (histogramNamesfileStructure == FileStructures.RUNDIRS):
                filePath = f"{directory}/{run}/{fileName}"
            else:
                raise Exception("Not a valid FileStructure!")
            with uproot.open(filePath) as file:
                for dirName in namesDict.keys():
                    dir = file[dirName]
                    for i, item in enumerate(dir):
                        if item.member("fName") in namesDict[dirName]:
                            fullName = dirName + '/' + item.member("fName")
                            tmpHist = dir[i]
                            tmpHistWritable = tmpHist.to_writable()
                            tmpHistPr = tmpHistWritable.to_pyroot()
                            if fullName not in histograms:
                                histograms[fullName] = tmpHistPr
                            else:
                                histograms[fullName].Add(tmpHistPr)

    elif depth == 1:
        for irun, run in enumerate(runList):
            print(f"Processing run {run} ({irun+1}/{len(runList)})...            ", end='\r')
            if (histogramNamesfileStructure == FileStructures.FLAT):
                filePath = f"{directory}/AnalysisResults_run{run}.root"
            elif (histogramNamesfileStructure == FileStructures.RUNDIRS):
                filePath = f"{directory}/{run}/{fileName}"
            else:
                raise Exception("Not a valid FileStructure!")
            with uproot.open(filePath) as file:
                for fullName in namesDict.keys():
                    tmpHist = file[fullName]
                    tmpHistWritable = tmpHist.to_writable()
                    tmpHistPr = tmpHistWritable.to_pyroot()
                    if fullName not in histograms:
                        histograms[fullName] = tmpHistPr
                    else:
                        # This is a special case for lumi histograms, which each have their own axis labels per run.
                        # ROOT will use TH1.Merge instead, this seems to give the expected outcome
                        histograms[fullName].Add(tmpHistPr)

    print("\nDone!")
    return histograms

def build_histograms_map(filePath):
    histogramsMap = {}
    print(f"Using {filePath} to create map of histograms")
    with uproot.open(filePath) as file:
        for dirName in [key for key in file.keys() if "output" in key]:
            dir = file[dirName]
            for i, item in enumerate(dir):
                for ii, iitem in enumerate(dir[i]):
                    fullName = dirName + '/' + item.member("fName") + '/' + iitem.member("fName")
                    histogramsMap[fullName] = {'dir': dirName, 'group': i, 'hist': ii}

    return histogramsMap

def get_histograms_from_file(filePath, fullHistogramNames, outputFormat='pyroot'):
    splitHistNames = [s.split("/") for s in fullHistogramNames] 
    namesDict = {}
    for irow, (row, fullName) in enumerate(zip(splitHistNames, fullHistogramNames)):
        depth = len(row)
        if row[1] == 'output;1' or row[1] == 'Statistics;1':
            depth -= 1
            row[0] += ('/' + row[1])
            row = np.delete(row, 1)
        elif row[0] == 'eventselection-run3' and ('luminosity' in row[1] or 'bcselection' in row[1]):
            # Special case: lumi hist can be accessed directly
            depth = 1
            namesDict[fullName] = fullName
        if depth == 3:
            if row[0] not in namesDict:
                namesDict[row[0]] = {row[1] : []}
            elif row[1] not in namesDict[row[0]]:
                namesDict[row[0]][row[1]] = []
            namesDict[row[0]][row[1]].append(row[2])
        elif depth == 2: # e. g. the lumi histogram
            if row[0] not in namesDict:
                namesDict[row[0]] = []
            namesDict[row[0]].append(row[1])
        elif depth == 1:
            pass
        else:
            raise Exception(f"Histogram name with depth = {depth}, don't know what to do!")

    histograms = {}
    if depth == 3:
        with uproot.open(filePath) as file:
            for dirName in namesDict.keys():
                try:
                    dir = file[dirName]
                except:
                    raise Exception(f"Couldn't get '{dirName}' from file '{filePath}'!")
                for i, item in enumerate(dir):
                    if item.member("fName") in namesDict[dirName]:
                        for ii, iitem in enumerate(dir[i]):
                            if iitem.member("fName") in namesDict[dirName][item.member("fName")]:
                                fullName = dirName + '/' + item.member("fName") + '/' + iitem.member("fName")
                                tmpHist = dir[i][ii]
                                tmpHistWritable = tmpHist.to_writable()
                                if outputFormat == 'pyroot':
                                    tmpHistOut = tmpHistWritable.to_pyroot()
                                elif outputFormat == 'hist':
                                    tmpHistOut = tmpHistWritable.to_hist()
                                else:
                                    raise Exception("outputFormat not recognized!")
                                histograms[fullName] = tmpHistOut

    elif depth == 2:
            with uproot.open(filePath) as file:
                for dirName in namesDict.keys():
                    dir = file[dirName]
                    for i, item in enumerate(dir):
                        if item.member("fName") in namesDict[dirName]:
                            fullName = dirName + '/' + item.member("fName")
                            tmpHist = dir[i]
                            tmpHistWritable = tmpHist.to_writable()
                            if outputFormat == 'pyroot':
                                tmpHistOut = tmpHistWritable.to_pyroot()
                            elif outputFormat == 'hist':
                                tmpHistOut = tmpHistWritable.to_hist()
                            else:
                                raise Exception("outputFormat not recognized!")
                            histograms[fullName] = tmpHistOut

    elif depth == 1:
        with uproot.open(filePath) as file:
            for fullName in namesDict.keys():
                tmpHist = file[fullName]
                tmpHistWritable = tmpHist.to_writable()
                if outputFormat == 'pyroot':
                    tmpHistOut = tmpHistWritable.to_pyroot()
                elif outputFormat == 'hist':
                    tmpHistOut = tmpHistWritable.to_hist()
                else:
                    raise Exception("outputFormat not recognized!")
                if fullName not in histograms:
                    histograms[fullName] = tmpHistOut
                else:
                    # This is a special case for lumi histograms, which each have their own axis labels per run.
                    # ROOT will use TH1.Merge instead, this seems to give the expected outcome
                    histograms[fullName].Add(tmpHistOut)

    return histograms

def compare_efficiency_upcmode(numeratorString, denominatorString, numeratorTitle, denominatorTitle, numeratorFilepathNo, denominatorFilepathNo, numeratorFilepathYes, denominatorFilepathYes):
    """
    Compare efficiencies between event with and without ITS UPC Mode reconstruction flag
    """
    binsPt = [0., 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8, 3.2, 3.6, 4., 4.4, 4.8, 5.2, 5.6, 6., 8., 10., 12., 16., 20.]
    genHistName = 'MyMcPtYHisto'
    recHistName = 'Pt'

    fileYesNumerator = r.TFile.Open(numeratorFilepathYes)
    fileNoNumerator = r.TFile.Open(numeratorFilepathNo)
    fileYesDenominator = r.TFile.Open(denominatorFilepathYes)
    fileNoDenominator = r.TFile.Open(denominatorFilepathNo)

    # Check that the files are what we think
    testHist = fileYesNumerator.Get("analysis-event-selection/output").FindObject("Event_AfterCuts").FindObject("IsITSUPCMode")
    if testHist.GetMean() != 1:
        raise Exception(f"File {numeratorFilepathYes} is not purely IsITSUPCMode=True!")
    testHist = fileYesDenominator.Get("analysis-event-selection/output").FindObject("Event_AfterCuts").FindObject("IsITSUPCMode")
    if testHist.GetMean() != 1:
        raise Exception(f"File {denominatorFilepathYes} is not purely IsITSUPCMode=True!")
    testHist = fileNoNumerator.Get("analysis-event-selection/output").FindObject("Event_AfterCuts").FindObject("IsITSUPCMode")
    if testHist.GetMean() != 0:
        raise Exception(f"File {numeratorFilepathNo} is not purely IsITSUPCMode=False!")
    testHist = fileNoDenominator.Get("analysis-event-selection/output").FindObject("Event_AfterCuts").FindObject("IsITSUPCMode")
    if testHist.GetMean() != 0:
        raise Exception(f"File {denominatorFilepathNo} is not purely IsITSUPCMode=False!")

    # Obtain histograms
    if 'Gen' in numeratorString:
        histYesNumeratorPtY = fileYesNumerator.Get("analysis-asymmetric-pairing/output").FindObject(numeratorString).FindObject(genHistName)
        lowerYBin = histYesNumeratorPtY.GetYaxis().FindBin(-0.9)
        upperYBin = histYesNumeratorPtY.GetYaxis().FindBin(0.9) - 1
        histYesNumeratorPt = histYesNumeratorPtY.ProjectionX(f"yesProjPtMcGen_y_{-0.9}_{0.9}", lowerYBin, upperYBin)
        histNoNumeratorPtY = fileNoNumerator.Get("analysis-asymmetric-pairing/output").FindObject(numeratorString).FindObject(genHistName)
        lowerYBin = histNoNumeratorPtY.GetYaxis().FindBin(-0.9)
        upperYBin = histNoNumeratorPtY.GetYaxis().FindBin(0.9) - 1
        histNoNumeratorPt = histNoNumeratorPtY.ProjectionX(f"noProjPtMcGen_y_{-0.9}_{0.9}", lowerYBin, upperYBin)
    else:
        histYesNumeratorPt = fileYesNumerator.Get("analysis-asymmetric-pairing/output").FindObject(numeratorString).FindObject(recHistName)
        histNoNumeratorPt = fileNoNumerator.Get("analysis-asymmetric-pairing/output").FindObject(numeratorString).FindObject(recHistName)
    histYesNumeratorPt = histYesNumeratorPt.Rebin(len(binsPt) - 1, "histYesNumeratorPt", np.asarray(binsPt, 'd'))
    histNoNumeratorPt = histNoNumeratorPt.Rebin(len(binsPt) - 1, "histNoNumeratorPt", np.asarray(binsPt, 'd'))

    if 'Gen' in denominatorString:
        histYesDenominatorPtY = fileYesDenominator.Get("analysis-asymmetric-pairing/output").FindObject(denominatorString).FindObject(genHistName)
        lowerYBin = histYesDenominatorPtY.GetYaxis().FindBin(-0.9)
        upperYBin = histYesDenominatorPtY.GetYaxis().FindBin(0.9) - 1
        histYesDenominatorPt = histYesDenominatorPtY.ProjectionX(f"yesProjPtMcGen_y_{-0.9}_{0.9}", lowerYBin, upperYBin)
        histNoDenominatorPtY = fileNoDenominator.Get("analysis-asymmetric-pairing/output").FindObject(denominatorString).FindObject(genHistName)
        lowerYBin = histNoDenominatorPtY.GetYaxis().FindBin(-0.9)
        upperYBin = histNoDenominatorPtY.GetYaxis().FindBin(0.9) - 1
        histNoDenominatorPt = histNoDenominatorPtY.ProjectionX(f"noProjPtMcGen_y_{-0.9}_{0.9}", lowerYBin, upperYBin)
    else:
        histYesDenominatorPt = fileYesDenominator.Get("analysis-asymmetric-pairing/output").FindObject(denominatorString).FindObject(recHistName)
        histNoDenominatorPt = fileNoDenominator.Get("analysis-asymmetric-pairing/output").FindObject(denominatorString).FindObject(recHistName)
    histYesDenominatorPt = histYesDenominatorPt.Rebin(len(binsPt) - 1, "histYesDenominatorPt", np.asarray(binsPt, 'd'))
    histNoDenominatorPt = histNoDenominatorPt.Rebin(len(binsPt) - 1, "histNoDenominatorPt", np.asarray(binsPt, 'd'))
    
    effNo = Efficiency(numeratorTitle, denominatorTitle, histNoNumeratorPt, histNoDenominatorPt)
    effNo.take_ownership()
    effYes = Efficiency(numeratorTitle, denominatorTitle, histYesNumeratorPt, histYesDenominatorPt)
    effYes.take_ownership()
    
    # Plotting
    cCompareYesNo = r.TCanvas()
    r.SetOwnership(cCompareYesNo, False)
    cCompareYesNo.cd()
    effNo.histogram.Draw()
    effNo.histogram.SetStats(0)
    effNo.histogram.GetXaxis().SetTitle("p_{T} (Gev/c)")
    effNo.histogram.GetYaxis().SetTitle("Efficiency")
    effNo.histogram.GetYaxis().SetRangeUser(0, 1)
    effNo.histogram.SetLineColor(r.kBlue)
    effNo.replace_displayed_title()
    effYes.histogram.Draw('same')
    effYes.histogram.SetLineColor(r.kRed)
    lCompareYesNo = r.TLegend(0.6, 0.1, 0.9, 0.3)
    r.SetOwnership(lCompareYesNo, False)
    lCompareYesNo.AddEntry(effNo.histogram, "IsITSUPCMode=False")
    lCompareYesNo.AddEntry(effYes.histogram, "IsITSUPCMode=True")
    lCompareYesNo.Draw()
    cCompareYesNo.Draw()

    return effNo, effYes

def run_by_run_data_mc(variableNames, units, dataDir, mcDir, daughter, runListFull, runList1, runList2, text=False):
    """
    Plot the mean value of a variable vs run number, for data and MC.
    """
    if len(units) == 0:
        units = ['' for _ in range(len(variableNames))]
    if daughter != 'kaon' and daughter != 'pion':
        raise Exception("'daughter' argument must be either 'kaon' or 'pion'!")
    runListFull = sorted(runListFull, key=lambda run: 1*int(run) if run in runList1 else 10*int(run) if run in runList2 else 100*int(run))
    dictListDataMeans = {key: [] for key in variableNames}
    dictListMcMeans = {key: [] for key in variableNames}
    dictListDataMeanErrors = {key: [] for key in variableNames}
    dictListMcMeanErrors = {key: [] for key in variableNames}
    listDataColors = []
    listDataTextColors = []
    listMcColors = []
    listMcTextColors = []
    listLabelColors = []
    dictColors = {'mc1': 'xkcd:pale lime green', 'data1': 'xkcd:pale cyan', 'mc2': 'xkcd:pale olive green', 'data2': 'xkcd:pastel blue', 'mc3': 'lightgray', 'data3': 'gray'}
    dictTextColors = {'mc1': 'xkcd:frog green', 'data1': 'xkcd:bright sky blue', 'mc2': 'xkcd:forrest green', 'data2': 'xkcd:cerulean', 'mc3': 'lightgray', 'data3': 'gray'}
    for irun, run in enumerate(runListFull):
        dictDataHists = {}
        dictMcHists = {}
        print(f"Processing run {run} ({irun+1}/{len(runListFull)})...            ", end='\r')
        dataFilepath = f"{dataDir}/{run}/AnalysisResults.root"
        try:
            with uproot.open(dataFilepath) as dataTmpFile:
                tmpDir = dataTmpFile["analysis-track-selection/output;1"]
            listLabelColors.append('black')
        except:
            print(f"Run {run} missing from data directory")
            for variableName in variableNames:
                dictListDataMeans[variableName].append(0)
                dictListDataMeanErrors[variableName].append(0)
                dictListMcMeans[variableName].append(0)
                dictListMcMeanErrors[variableName].append(0)
            listDataColors.append('black')
            listDataTextColors.append('black')
            listMcColors.append('black')
            listMcTextColors.append('black')
            listLabelColors.append('red')
            continue
        for i, item in enumerate(tmpDir):
            if item.member("fName") == "TrackBarrel_MyStandardPrimaryTrackDCACut":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") in variableNames:
                        dictDataHists[iitem.member("fName")] = tmpDir[i][ii]

        for variableName in variableNames:
            try:
                dataHist = dictDataHists[variableName].to_numpy()
            except:
                print(f"Histogram '{variableName}' not found in data file")
                continue
            dataBinCenters = (dataHist[1][:-1] + dataHist[1][1:]) / 2
            dataMean = np.average(dataBinCenters, weights=dataHist[0])
            dictListDataMeans[variableName].append(dataMean)
            dataSampleStdDev = np.sqrt(np.average((dataBinCenters - dataMean)**2, weights=dataHist[0]))
            dataMeanError = dataSampleStdDev / np.sqrt(len(dataHist[0]))
            dictListDataMeanErrors[variableName].append(dataMeanError)

        mcFilePath = f"{mcDir}/AnalysisResults_run{run}.root"
        with uproot.open(mcFilePath) as mcTmpFile:
            tmpDir = mcTmpFile["analysis-track-selection/output;1"]
        for i, item in enumerate(tmpDir):
            if item.member("fName") == "AssocsBarrel_BeforeCuts":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") in variableNames:
                        dictMcHists[iitem.member("fName")] = tmpDir[i][ii]

        for variableName in variableNames:
            try:
                mcHist = dictMcHists[variableName].to_numpy()
            except:
                print(f"Histogram '{variableName}' not found in mc file")
                continue
            mcBinCenters = (mcHist[1][:-1] + mcHist[1][1:]) / 2
            mcMean = np.average(mcBinCenters, weights=mcHist[0])
            dictListMcMeans[variableName].append(mcMean)
            mcSampleStdDev = np.sqrt(np.average((mcBinCenters - mcMean)**2, weights=mcHist[0]))
            mcMeanError = mcSampleStdDev / np.sqrt(len(mcHist[0]))
            dictListMcMeanErrors[variableName].append(mcMeanError)

        if run in runList1:
            listDataColors.append(dictColors['data1'])
            listDataTextColors.append(dictTextColors['data1'])
            listMcColors.append(dictColors['mc1'])
            listMcTextColors.append(dictTextColors['mc1'])
        elif run in runList2:
            listDataColors.append(dictColors['data2'])
            listDataTextColors.append(dictTextColors['data2'])
            listMcColors.append(dictColors['mc2'])
            listMcTextColors.append(dictTextColors['mc2'])
        else:
            listDataColors.append(dictColors['data3'])
            listDataTextColors.append(dictTextColors['data3'])
            listMcColors.append(dictColors['mc3'])
            listMcTextColors.append(dictTextColors['mc3'])
    print("Complete!                        ")

    runAxis = np.arange(len(runListFull))
    fig, axs = plt.subplots(len(variableNames), 1, figsize=(20, 5 * len(variableNames)))
    width = 0.5

    for i, ax in enumerate(axs):
        ax.set_title(f"{variableNames[i]}, data primary tracks and MC reco {daughter}")
        barData = ax.bar(runAxis - width*0.5, dictListDataMeans[variableNames[i]], width, yerr=dictListDataMeanErrors[variableNames[i]], color=listDataColors, capsize=3)
        barMc = ax.bar(runAxis + width*0.5, dictListMcMeans[variableNames[i]], width, yerr=dictListMcMeanErrors[variableNames[i]], color=listMcColors, capsize=3)
        for j, label in enumerate(runListFull):
            if text:
                ax.text(runAxis[j] - 0.5, listMeanEff[j] + 0.05, f'{listMeanEff[j]:.4f}$\pm${listMeanEffError[j]:.4f}')
            ax.text(runAxis[j], ax.get_ylim()[0]-(0.008*(ax.get_ylim()[1]-ax.get_ylim()[0])), label, ha='center', va='top', rotation=90, color=listLabelColors[j])
        ax.set_xticks(runAxis, '', rotation=90)
        ax.set_ylabel(f'$\langle${variableNames[i]}$\\rangle$' + ('' if units[i]=='' else f' ({units[i]})'))
        patchData = mpatches.Patch(color=dictColors['data2'], label='Data')
        patchMc = mpatches.Patch(color=dictColors['mc2'], label='Mc')
        ax.legend(handles=[patchData, patchMc])
    plt.tight_layout()
    plt.show()

    return axs

def run_by_run_num_candidates_compare_runlists(dataDir, mcDir, runListFull, runList1, runList2, mmin=1.81, mmax=1.90, text=False, useLumi=True):
    """
    Plot the number of D0 candidates vs run number, for data and MC.
    """
    runListFull = sorted(runListFull, key=lambda run: 1*int(run) if run in runList1 else 10*int(run) if run in runList2 else 100*int(run))
    listDataNumCandidates = []
    listMcNumCandidates = []
    dataNumCandidatesTotal = 0
    mcNumCandidatesTotal = 0
    listDataColors = []
    listDataTextColors = []
    listMcColors = []
    listMcTextColors = []
    listLabelColors = []
    dictColors = {'mc1': 'xkcd:pale lime green', 'data1': 'xkcd:pale cyan', 'mc2': 'xkcd:pale olive green', 'data2': 'xkcd:pastel blue', 'mc3': 'lightgray', 'data3': 'gray'}
    dictTextColors = {'mc1': 'xkcd:frog green', 'data1': 'xkcd:bright sky blue', 'mc2': 'xkcd:forrest green', 'data2': 'xkcd:cerulean', 'mc3': 'lightgray', 'data3': 'gray'}

    if useLumi:
        lumiFile = r.TFile.Open("~/cernbox/singlegap/LHC23_PbPb_pass4_train355821/mergedAnalysisResults.root")
        lumiHist = lumiFile.Get("bc-selection-task").Get("hLumiTCEafterBCcuts")
        for run in runListFull:
            try:
                lumi = lumiHist.GetBinContent(lumiHist.GetXaxis().FindBin(run))
                listDataNumCandidates.append(lumi)
                dataNumCandidatesTotal += lumi
                listLabelColors.append('black')
            except:
                print(f"Run {run} missing from data directory")
                listDataColors.append('black')
                listDataTextColors.append('black')
                listMcColors.append('black')
                listMcTextColors.append('black')
                listLabelColors.append('red')
                continue

    for irun, run in enumerate(runListFull):
        print(f"Processing run {run} ({irun+1}/{len(runListFull)})...            ", end='\r')
        if not useLumi:
            if 'perlmutter' in dataDir:
                dataFilepath = f"{dataDir}/AnalysisResults_run{run}.root"
            else:
                dataFilepath = f"{dataDir}/{run}/AnalysisResults.root"
            try:
                with uproot.open(dataFilepath) as dataTmpFile:
                    tmpDir = dataTmpFile["analysis-asymmetric-pairing/output;1"]
                listLabelColors.append('black')
            except:
                print(f"Run {run} missing from data directory")
                listDataColors.append('black')
                listDataTextColors.append('black')
                listMcColors.append('black')
                listMcTextColors.append('black')
                listLabelColors.append('red')
                continue
            for i, item in enumerate(tmpDir):
                # TODO: Which cuts should be applied when counting these candidates?
                if item.member("fName") == "PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepTauxyzprojCut":
                    for ii, iitem in enumerate(tmpDir[i]):
                        if iitem.member("fName") == 'Mass':
                            dataHistRaw = tmpDir[i][ii]

            dataHist = dataHistRaw.to_hist()
            dataNumCandidates = dataHist[complex(0, mmin):complex(0, mmax)].sum().value
            listDataNumCandidates.append(dataNumCandidates)
            dataNumCandidatesTotal += dataNumCandidates

        mcFilePath = f"{mcDir}/AnalysisResults_run{run}.root"
        with uproot.open(mcFilePath) as mcTmpFile:
            tmpDir = mcTmpFile["analysis-asymmetric-pairing/output;1"]
        for i, item in enumerate(tmpDir):
            # The total number of generator lvl signal is the relevant number for weighing of the sum of efficiecies over runs
            if item.member("fName") == "MCTruthGenAfterBcCuts_D0FS":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'MyMcPtYHisto':
                        mcHistRaw = tmpDir[i][ii]

        mcHist = mcHistRaw.to_hist()
        mcNumCandidates = mcHist[:,-0.9j:0.9j].sum().value
        listMcNumCandidates.append(mcNumCandidates)
        mcNumCandidatesTotal += mcNumCandidates

        if run in runList1:
            listDataColors.append(dictColors['data1'])
            listDataTextColors.append(dictTextColors['data1'])
            listMcColors.append(dictColors['mc1'])
            listMcTextColors.append(dictTextColors['mc1'])
        elif run in runList2:
            listDataColors.append(dictColors['data2'])
            listDataTextColors.append(dictTextColors['data2'])
            listMcColors.append(dictColors['mc2'])
            listMcTextColors.append(dictTextColors['mc2'])
        else:
            listDataColors.append(dictColors['data3'])
            listDataTextColors.append(dictTextColors['data3'])
            listMcColors.append(dictColors['mc3'])
            listMcTextColors.append(dictTextColors['mc3'])
    print("Complete!                        ")

    # Normalize by total number of candidates -- for efficiency it is the distribution between run that matters, not the absolute values
    listDataNumCandidates = np.array(listDataNumCandidates) / dataNumCandidatesTotal
    listMcNumCandidates = np.array(listMcNumCandidates) / mcNumCandidatesTotal

    runAxis = np.arange(len(runListFull))
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    width = 0.4

    if not useLumi:
        ax.set_title(f"D0 candidates in data (pairs with {mmin} < m < {mmax} GeV/c2) and MC generated D0 (after BC cuts)")
    else:
        ax.set_title(f"Lumi in data and MC generated D0 (after BC cuts)")
    barData = ax.bar(runAxis - width*0.5, listDataNumCandidates, width, color=listDataColors, capsize=3)
    barMc = ax.bar(runAxis + width*0.5, listMcNumCandidates, width, color=listMcColors, capsize=3)
    for j, label in enumerate(runListFull):
        ax.text(runAxis[j], ax.get_ylim()[0]-(0.008*(ax.get_ylim()[1]-ax.get_ylim()[0])), label, ha='center', va='top', rotation=90, color=listLabelColors[j])
    ax.set_xticks(runAxis, '', rotation=90)
    ax.set_ylabel("Fraction of total")
    patchData = mpatches.Patch(color=dictColors['data2'], label='Data' + (' lumi' if useLumi else ''))
    patchMc = mpatches.Patch(color=dictColors['mc2'], label='Mc')
    ax.legend(handles=[patchData, patchMc])
    plt.tight_layout()
    plt.xlim([-2*width, runAxis.size])
    plt.show()

    return ax, listDataNumCandidates, listMcNumCandidates

def run_by_run_efficiency_compare_runlists(numeratorString, denominatorString, numeratorTitle, denominatorTitle, numeratorDir, denominatorDir, runListFull, runList1, runList2, minPt=0., maxPt=20., text=False, doPlot=True):
    """
    Given a directory of AnalysisResults.root files for each run, create a plot showing a given efficiency vs run number. Use uproot for I/O.
    """
    if 'Gen' in numeratorString:
        numeratorHistName = 'MyMcPtYHisto'
    else:
        numeratorHistName = 'Pt'
    if 'Gen' in denominatorString:
        denominatorHistName = 'MyMcPtYHisto'
    else:
        denominatorHistName = 'Pt'

    runListFull = sorted(runListFull, key=lambda run: 1*int(run) if run in runList1 else 10*int(run) if run in runList2 else 100*int(run))
    listMeanEff = []
    listMeanEffError = []
    totalNumCountsRunList1 = 0
    totalNumCountsRunList2 = 0
    totalDenCountsRunList1 = 0
    totalDenCountsRunList2 = 0
    listColors = []
    listTextColors = []
    dictColors = {'1': 'xkcd:baby blue', '2': 'xkcd:pale pink', '3': 'lightgray'}
    dictTextColors = {'1': 'xkcd:bright blue', '2': 'xkcd:red', '3': 'lightgray'}
    for irun, run in enumerate(runListFull):
        print(f"Processing run {run} ({irun+1}/{len(runListFull)})...            ", end='\r')
        numeratorFilepath = f"{numeratorDir}/AnalysisResults_run{run}.root"
        with uproot.open(numeratorFilepath) as numTmpFile:
            tmpDir = numTmpFile["analysis-asymmetric-pairing/output;1"]
        for i, item in enumerate(tmpDir):
            if item.member("fName") == numeratorString:
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == numeratorHistName:
                        numHistRaw = tmpDir[i][ii]
        numHist = numHistRaw.to_hist()
        # If we are looking for a generator lvl histogram, we need to project out out rapidity range
        if 'Gen' in numeratorString:
            numCount = numHist[complex(0, minPt):complex(0, maxPt),-0.9j:0.9j].sum().value
        else:
            numCount = numHist[complex(0, minPt):complex(0, maxPt)].sum().value

        denominatorFilepath = f"{denominatorDir}/AnalysisResults_run{run}.root"
        with uproot.open(denominatorFilepath) as denTmpFile:
            tmpDir = denTmpFile["analysis-asymmetric-pairing/output;1"]
        for i, item in enumerate(tmpDir):
            if item.member("fName") == denominatorString:
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == denominatorHistName:
                        denHistRaw = tmpDir[i][ii]
        denHist = denHistRaw.to_hist()
        # If we are looking for a generator lvl histogram, we need to project out out rapidity range
        if 'Gen' in denominatorString:
            denCount = denHist[complex(0, minPt):complex(0, maxPt),-0.9j:0.9j].sum().value
        else:
            denCount = denHist[complex(0, minPt):complex(0, maxPt)].sum().value

        # We want the average efficiency per run; rebin such that we have only one bin, extract the content and error
        if numCount == 0:
            eff = 0
            effError = 0
        else:
            eff = numCount / denCount
            effError = (1/denCount) * np.sqrt(numCount * (1 - numCount/denCount)) # Binomial error calculation
        listMeanEff.append(eff)
        listMeanEffError.append(effError)
        
        if run in runList1:
            listColors.append(dictColors['1'])
            listTextColors.append(dictTextColors['1'])
            totalNumCountsRunList1 += numCount
            totalDenCountsRunList1 += denCount
        elif run in runList2:
            listColors.append(dictColors['2'])
            listTextColors.append(dictTextColors['2'])
            totalNumCountsRunList2 += numCount
            totalDenCountsRunList2 += denCount
        else:
            listColors.append(dictColors['3'])
            listTextColors.append(dictTextColors['3'])
    print("Complete!                        ")

    totalNumErrRunList1 = np.sqrt(totalNumCountsRunList1)
    totalDenErrRunList1 = np.sqrt(totalDenCountsRunList1)
    try:
        meanEffRunList1 = totalNumCountsRunList1 / totalDenCountsRunList1
    except:
        meanEffRunList1 = 0
    try:
        meanEffErrRunList1 = (1/totalDenCountsRunList1) * np.sqrt(totalNumCountsRunList1 * (1 - totalNumCountsRunList1/totalDenCountsRunList1)) # Binomial error calculation
    except:
        meanEffErrRunList1 = 0

    totalNumErrRunList2 = np.sqrt(totalNumCountsRunList2)
    totalDenErrRunList2 = np.sqrt(totalDenCountsRunList2)
    try:
        meanEffRunList2 = totalNumCountsRunList2 / totalDenCountsRunList2
    except:
        meanEffRunList2 = 0
    try:
        meanEffErrRunList2 = (1/totalDenCountsRunList2) * np.sqrt(totalNumCountsRunList2 * (1 - totalNumCountsRunList2/totalDenCountsRunList2)) # Binomial error calculation
    except:
        meanEffErrRunList2 = 0
        
    runAxis = np.arange(len(runListFull))
    title = f"$\\frac{{\\text{{{numeratorTitle}}}}}{{\\text{{{denominatorTitle}}}}}$, {minPt} < $p_\\text{{T}}$ < {maxPt} GeV/c"
    if doPlot:
        plt.figure(figsize=(15, 5))
        plt.title(title)
        plt.bar(runAxis, listMeanEff, yerr=listMeanEffError, color=listColors, capsize=3)
        # plt.hlines(meanEffRunList1, -0.5, len(runList1)-0.5, alpha=0.5, color=dictTextColors['1'])
        # plt.hlines(meanEffRunList2, len(runList1)-0.5, len(runList1)+len(runList2)-0.5, alpha=0.5, color=dictTextColors['2'])
        plt.fill_between(runAxis, meanEffRunList1 - meanEffErrRunList1, meanEffRunList1 + meanEffErrRunList1, where=(runAxis > -1.) & (runAxis < len(runList1)), color=dictTextColors['1'], alpha=0.5)
        plt.fill_between(runAxis, meanEffRunList2 - meanEffErrRunList2, meanEffRunList2 + meanEffErrRunList2, where=(runAxis > len(runList1)-1.) & (runAxis < len(runList1)+len(runList2)), color=dictTextColors['2'], alpha=0.5)
        for i, label in enumerate(runListFull):
            if text:
                plt.text(runAxis[i] - 0.5, listMeanEff[i] + 0.05, f'{listMeanEff[i]:.4f}$\pm${listMeanEffError[i]:.4f}')
            plt.text(runAxis[i], plt.ylim()[0]-(0.008*(plt.ylim()[1]-plt.ylim()[0])), label, ha='center', va='top', color=listTextColors[i], rotation=90)
        plt.xticks(runAxis, '', rotation=90)
        plt.ylabel('$\\langle\\epsilon\\rangle$')
        plt.tight_layout()
        plt.show()
    return runAxis, listMeanEff, listMeanEffError, title

def run_by_run_efficiency(numeratorString, denominatorString, numeratorTitle, denominatorTitle, runList, numeratorDir, denominatorDir=None, minPt=0., maxPt=20., minY=-0.8, maxY=0.8, irFilePath='~/cernbox/notebooks/pyD0/interactionRate.txt', text=False):
    """
    Given a directory of AnalysisResults.root files for each run, create a plot showing a given efficiency vs interaction rate. Use uproot for I/O.
    """
    if denominatorDir is None:
        denominatorDir = numeratorDir
    if 'Gen' in numeratorString:
        numeratorHistName = 'MyMcPtYHisto'
    else:
        numeratorHistName = 'Y_PtFine'
    if 'Gen' in denominatorString:
        denominatorHistName = 'MyMcPtYHisto'
    else:
        denominatorHistName = 'Y_PtFine'

    df = pd.read_csv(irFilePath, sep=' ', names=['runNumber', 'ir'])
    df['ir'] = df['ir'].astype(float)
    df['runNumber'] = df['runNumber'].astype(str)
    df = df[df['runNumber'].isin(runList)]
    df = df.set_index('runNumber')

    for irun, run in enumerate(runList):
        print(f"Processing run {run} ({irun+1}/{len(runList)})...            ", end='\r')
        numeratorFilepath = f"{numeratorDir}/AnalysisResults_run{run}.root"
        with uproot.open(numeratorFilepath) as numTmpFile:
            tmpDir = numTmpFile["analysis-asymmetric-pairing/output;1"]
        for i, item in enumerate(tmpDir):
            if item.member("fName") == numeratorString:
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == numeratorHistName:
                        numHistRaw = tmpDir[i][ii]
        numHist = numHistRaw.to_hist()
        # Project out our pT and rapidity range
        if 'Gen' in numeratorString:
            numCount = numHist[complex(0, minPt):complex(0, maxPt),complex(0, minY):complex(0, maxY)].sum().value
        else:
            numCount = numHist[complex(0, minY):complex(0, maxY),complex(0, minPt):complex(0, maxPt)].sum().value

        denominatorFilepath = f"{denominatorDir}/AnalysisResults_run{run}.root"
        with uproot.open(denominatorFilepath) as denTmpFile:
            tmpDir = denTmpFile["analysis-asymmetric-pairing/output;1"]
        for i, item in enumerate(tmpDir):
            if item.member("fName") == denominatorString:
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == denominatorHistName:
                        denHistRaw = tmpDir[i][ii]
        denHist = denHistRaw.to_hist()
        # Project out our pT and rapidity range
        if 'Gen' in denominatorString:
            denCount = denHist[complex(0, minPt):complex(0, maxPt),complex(0, minY):complex(0, maxY)].sum().value
        else:
            denCount = denHist[complex(0, minY):complex(0, maxY),complex(0, minPt):complex(0, maxPt)].sum().value

        # We want the average efficiency per run; rebin such that we have only one bin, extract the content and error
        if numCount == 0:
            eff = 0
            effError = 0
        else:
            eff = numCount / denCount
            effError = (1/denCount) * np.sqrt(numCount * (1 - numCount/denCount)) # Binomial error calculation
        df.loc[run, 'eff'] = eff
        df.loc[run, 'effError'] = effError
        
    print("Complete!                        ")

    title = f"$\\frac{{\\text{{{numeratorTitle}}}}}{{\\text{{{denominatorTitle}}}}}$, {minPt} < $p_\\text{{T}}$ < {maxPt} GeV/c, {minY} < y < {maxY}"
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title(title)
    ax.errorbar(df['ir'], df['eff'], yerr=df['effError'], capsize=3, fmt="o")
    ax.set_ylabel('$\\epsilon$')
    plt.tight_layout()
    plt.show()
    return df

def run_by_run_num_candidates(dataDir, runList, lumiFilePath="~/cernbox/singlegap/LHC23_PbPb_pass5_train590144/mergedAnalysisResults_good.root",
                              mmin=1.81, mmax=1.90, minPt=0., maxPt=12., minY=-0.8, maxY=0.8, text=False, irFilePath='~/cernbox/notebooks/pyD0/interactionRate.txt'):
    """
    Plot the number of D0 candidates vs run interaction rate
    """

    df = pd.read_csv(irFilePath, sep=' ', names=['runNumber', 'ir'])
    df['ir'] = df['ir'].astype(float)
    df['runNumber'] = df['runNumber'].astype(str)
    df = df[df['runNumber'].isin(runList)]
    df = df.set_index('runNumber')

    lumiFile = r.TFile.Open(lumiFilePath)
    lumiHist = lumiFile.Get('eventselection-run3').Get('luminosity').Get('hLumiTCEafterBCcuts')
    for run in runList:
        lumi = lumiHist.GetBinContent(lumiHist.GetXaxis().FindBin(run))
        df.loc[run, 'lumi'] = lumi

    for irun, run in enumerate(runList):
        print(f"Processing run {run} ({irun+1}/{len(runList)})...            ", end='\r')
        if 'perlmutter' in dataDir:
            dataFilepath = f"{dataDir}/AnalysisResults_run{run}.root"
        else:
            dataFilepath = f"{dataDir}/{run}/AnalysisResults.root"
        with uproot.open(dataFilepath) as dataTmpFile:
            tmpDir = dataTmpFile["analysis-asymmetric-pairing/output;1"]
        for i, item in enumerate(tmpDir):
            # TODO: Which cuts should be applied when counting these candidates?
            if item.member("fName") == "PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepLxyCosPointingAngleCut":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'MyMassPtYHisto':
                        dataHistRaw = tmpDir[i][ii]

        dataHist = dataHistRaw.to_hist()
        dataNumCandidates = dataHist[complex(0, mmin):complex(0, mmax), complex(0, minPt):complex(0, maxPt), complex(0, minY):complex(0, maxY)].sum().value
        df.loc[run, 'dataNumCandidates'] = dataNumCandidates

    print("Complete!                        ")

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.set_title(f"D0 candidates / lumi (µb) in data ({mmin} < m < {mmax} GeV/c2, {minPt} < pT < {maxPt} GeV/c, {minY} < y < {maxY})")
    ax.scatter(df['ir'], df['dataNumCandidates']/df['lumi'])
    plt.tight_layout()
    plt.show()

    return df

def get_run_by_run_mc_info(mcDir, runList, lumiFilePath="/media/sigurd/T7/analysis/data/LHC23_PbPb_pass5_train590144/mergedAnalysisResults_good.root",
                           mmin=1.81, mmax=1.90, minPt=0., maxPt=12., minY=-0.8, maxY=0.8, text=False, irFilePath='~/cernbox/notebooks/pyD0/interactionRate.txt', fileStructure=FileStructures.RUNDIRS):
    """
    Create a dataframe containing run-by-run quantities for mc
    """

    df = pd.read_csv(irFilePath, sep=' ', names=['runNumber', 'ir'])
    df['ir'] = df['ir'].astype(float)
    df['runNumber'] = df['runNumber'].astype(str)
    df = df[df['runNumber'].isin(runList)]
    df = df.set_index('runNumber')

    lumiFile = r.TFile.Open(lumiFilePath)
    lumiHist = lumiFile.Get('eventselection-run3').Get('luminosity').Get('hLumiTCEafterBCcuts')
    for run in runList:
        lumi = lumiHist.GetBinContent(lumiHist.GetXaxis().FindBin(run))
        df.loc[run, 'lumi'] = lumi

    for irun, run in enumerate(runList):
        print(f"Processing run {run} ({irun+1}/{len(runList)})...            ", end='\r')
        if fileStructure is FileStructures.FLAT:
            mcFilepath = f"{mcDir}/AnalysisResults_run{run}.root"
        else:
            mcFilepath = f"{mcDir}/{run}/AnalysisResults.root"

        # Get event info
        histRawIsITSUPCMode = None
        histRawMcBcInTF = None
        with uproot.open(mcFilepath) as mcTmpFile:
            tmpDir = mcTmpFile["analysis-event-selection/output;1"]
        for i, item in enumerate(tmpDir):
            # TODO: Which cuts should be applied when counting these candidates?
            if item.member("fName") == "Event_BeforeCuts":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'VtxNContribReal':
                        histRawVtxNContribReal = tmpDir[i][ii]
                    elif iitem.member("fName") == 'VtxNContrib':
                        histRawVtxNContrib = tmpDir[i][ii]
                    elif iitem.member("fName") == 'MultNTracksPVeta1':
                        histRawMultNTracksPVeta1 = tmpDir[i][ii]
                    elif iitem.member("fName") == 'MyIsITSUPCModeHisto':
                        histRawIsITSUPCMode = tmpDir[i][ii]
            elif item.member("fName") == "Event_AfterCuts":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'VtxNContribReal':
                        histRawAfterCutsVtxNContribReal = tmpDir[i][ii]
            elif item.member("fName") == "EventsMC":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'MyMcIsNoITSROFBorderMcIsNoTFBorderHisto' or iitem.member("fName") == 'MyMcIsNoITSROFBorderRecomputedMcIsNoTFBorderRecomputedHisto':
                        histRawMcIsBorder2d = tmpDir[i][ii]
                    elif iitem.member("fName") == 'MyMCBcInTFHisto':
                        histRawMcBcInTF = tmpDir[i][ii]

        histVtxNContribReal = histRawVtxNContribReal.to_writable().to_pyroot()
        df.loc[run, 'meanVtxNContribReal'] = histVtxNContribReal.GetMean()
        df.loc[run, 'meanVtxNContribRealError'] = histVtxNContribReal.GetMeanError()
        df.loc[run, 'nEventsVtxNContribRealUnder17'] = histVtxNContribReal.Integral(1,17)
        df.loc[run, 'nEventsVtxNContribRealOver16'] = histVtxNContribReal.Integral(18,-1)
        # Count the number of events before cuts (but after TF and ROF border cuts) in this run
        df.loc[run, 'nEventsBeforeCuts'] = histVtxNContribReal.GetEntries()
        histVtxNContrib = histRawVtxNContrib.to_writable().to_pyroot()
        df.loc[run, 'meanVtxNContrib'] = histVtxNContrib.GetMean()
        df.loc[run, 'meanVtxNContribError'] = histVtxNContrib.GetMeanError()
        df.loc[run, 'nEventsVtxNContribUnder17'] = histVtxNContrib.Integral(1,17)
        df.loc[run, 'nEventsVtxNContribOver16'] = histVtxNContrib.Integral(18,-1)
        df.loc[run, 'meanMultNTracksPVeta1'] = histRawMultNTracksPVeta1.to_writable().to_pyroot().GetMean()
        # Count the number of events reconstructed with UPC mode
        if histRawIsITSUPCMode is not None:
            histIsITSUPCMode = histRawIsITSUPCMode.to_writable().to_pyroot()
            if histIsITSUPCMode.Integral() < histIsITSUPCMode.GetEntries():
                print(f"WARNING: Run {run} IsITSUPCMode has {histIsITSUPCMode.GetEntries()} entries but the integral is {histIsITSUPCMode.Integral()}!")
                print(f"Bin 1 content = {histIsITSUPCMode.GetBinContent(1)}")
                print(f"Bin 2 content = {histIsITSUPCMode.GetBinContent(2)}")
            df.loc[run, 'nEventsBeforeCutsITSUPCModeFalse'] = histIsITSUPCMode.GetBinContent(1)
            df.loc[run, 'nEventsBeforeCutsITSUPCModeTrue'] = histIsITSUPCMode.GetBinContent(2)
        # Count the number of events after cuts in this run
        df.loc[run, 'nEventsAfterCuts'] = histRawAfterCutsVtxNContribReal.to_writable().to_pyroot().GetEntries()
        # Count the number of MC events after MC TF and ROF border cuts in this run
        df.loc[run, 'nEventsMC'] = histRawMcIsBorder2d.to_writable().to_pyroot().GetBinContent(2,2)
        # If present, use the MC BC in TF histogram to figure out the number of BCs and orbits in the TF
        if histRawMcBcInTF is not None:
            histMcBcInTF = histRawMcBcInTF.to_hist()
            nz = np.nonzero(histMcBcInTF.values())[0]
            df.loc[run, 'highestMCBcInTF'] = histMcBcInTF.axes[0].centers[nz[-1]]

        # Get TF and other table-maker level info
        histRawMcGenId = None
        histRawRecGenId = None
        histRawTF_NMCCollisions = None
        histRawTF_NCollisions = None
        with uproot.open(mcFilepath) as mcTmpFile:
            tmpDir = mcTmpFile["table-maker-m-c/output;1"]
        for i, item in enumerate(tmpDir):
            # TODO: Which cuts should be applied when counting these candidates?
            if item.member("fName") == "TimeFrameStats":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'TF_NMCCollisions':
                        histRawTF_NMCCollisions = tmpDir[i][ii]
                    elif iitem.member("fName") == 'TF_NCollisions':
                        histRawTF_NCollisions = tmpDir[i][ii]
            elif item.member("fName") == "Event_MCTruth": 
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'MultMCNParticlesEta10':
                        histRawMcNParticlesEta10 = tmpDir[i][ii]
                    elif iitem.member("fName") == 'MyGenIdHisto':
                        histRawMcGenId = tmpDir[i][ii]
            elif item.member("fName") == "Event_BeforeCuts": # To get apples-to-apples comparison to Event_MCTruth, we need histograms before BC cuts
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'MyGenIdHisto':
                        histRawRecGenId = tmpDir[i][ii]

        if histRawTF_NMCCollisions is not None:
            histTF_NMCCollisions = histRawTF_NMCCollisions.to_writable().to_pyroot()
            df.loc[run, 'meanTFNMCCollisions'] = histTF_NMCCollisions.GetMean()
            df.loc[run, 'stdDevTFNMCCollisions'] = histTF_NMCCollisions.GetStdDev()
        if histRawTF_NCollisions is not None:
            histTF_NCollisions = histRawTF_NCollisions.to_writable().to_pyroot()
            df.loc[run, 'meanTFNCollisions'] = histTF_NCollisions.GetMean()
            df.loc[run, 'stdDevTFNCollisions'] = histTF_NCollisions.GetStdDev()
            df.loc[run, 'meanMultMcNParticlesEta10'] = histRawMcNParticlesEta10.to_writable().to_pyroot().GetMean()
        if histRawRecGenId is not None and histRawMcGenId is not None:
            histMcGenId = histRawMcGenId.to_writable().to_hist()
            values = histMcGenId.values()
            mask = values != 0
            nonzero_centers = histMcGenId.axes[0].centers[mask]
            nonzero_values = values[mask]
            for x, c in zip(nonzero_centers, nonzero_values):
                df.loc[run, f'nMcEventsGenId_{int(x)}'] = c
            histRecGenId = histRawRecGenId.to_writable().to_hist()
            values = histRecGenId.values()
            mask = values != 0
            nonzero_centers = histRecGenId.axes[0].centers[mask]
            nonzero_values = values[mask]
            for x, c in zip(nonzero_centers, nonzero_values):
                df.loc[run, f'nRecEventsGenId_{int(x)}'] = c

        # Temporarily remove this until I have valid AnalysisResults files again
        """
        # Get pair info
        with uproot.open(mcFilepath) as mcTmpFile:
            tmpDir = mcTmpFile["analysis-asymmetric-pairing/output;1"]
        for i, item in enumerate(tmpDir):
            # TODO: Which cuts should be applied when counting these candidates?
            if item.member("fName") == "PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_singleGapTrackCuts4_PtDepLxyCosPointingAngleCut_KPiFromD0FS":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'Y_PtFine':
                        mcHistRecRaw = tmpDir[i][ii]
            elif item.member("fName") == "MCTruthGenAfterBcCuts_D0FS":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'MyMcPtYHisto':
                        mcHistGenRaw = tmpDir[i][ii]

        mcHistRec = mcHistRecRaw.to_hist()
        mcNumRecCandidates = mcHistRec[complex(0, minY):complex(0, maxY), complex(0, minPt):complex(0, maxPt)].sum().value
        df.loc[run, 'mcNumRecCandidates'] = mcNumRecCandidates
        mcHistGen = mcHistGenRaw.to_hist()
        mcNumGenCandidates = mcHistGen[complex(0, minPt):complex(0, maxPt), complex(0, minY):complex(0, maxY)].sum().value
        df.loc[run, 'mcNumGenCandidates'] = mcNumGenCandidates
        """

    print("Complete!                        ")

    return df

def get_run_by_run_tablemaker_info(dir, runList, irFilePath='~/cernbox/notebooks/pyD0/interactionRate.txt'):
    df = pd.read_csv(irFilePath, sep=' ', names=['runNumber', 'ir'])
    df['ir'] = df['ir'].astype(float)
    df['runNumber'] = df['runNumber'].astype(str)
    df = df[df['runNumber'].isin(runList)]
    df = df.set_index('runNumber')

    for irun, run in enumerate(runList):
        print(f"Processing run {run} ({irun+1}/{len(runList)})...            ", end='\r')
        dataFilepath = f"{dir}/{run}/mergedAnalysisResults.root"

        with uproot.open(dataFilepath) as dataTmpFile:
            tmpDir = dataTmpFile["table-maker/output;1"]
        for i, item in enumerate(tmpDir):
            if item.member("fName") == "Event_BeforeCuts":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'VtxNContrib':
                        histRawVtxNContribBeforeCuts = tmpDir[i][ii]
            if item.member("fName") == "Event_AfterCuts":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'VtxNContrib':
                        histRawVtxNContribAfterCuts = tmpDir[i][ii]

        with uproot.open(dataFilepath) as file:
            histRawLumi = file["eventselection-run3/luminosity;1/hLumiTCE"]
            histRawColCounterAcc = file["eventselection-run3/eventselection;1/hColCounterAcc"]
        histLumi = histRawLumi.to_writable().to_pyroot()
        df.loc[run, 'lumiTCE'] = histLumi.GetEntries()
        histColCounterAcc = histRawColCounterAcc.to_writable().to_pyroot()
        df.loc[run, 'colCounterAcc'] = histColCounterAcc.GetEntries()

        histVtxNContribBeforeCuts = histRawVtxNContribBeforeCuts.to_writable().to_pyroot()
        df.loc[run, 'meanVtxNContribBeforeCuts'] = histVtxNContribBeforeCuts.GetMean()
        df.loc[run, 'nEventsVtxNContribUnder16BeforeCuts'] = histVtxNContribBeforeCuts.Integral(1,16)
        df.loc[run, 'nEventsVtxNContribOver15BeforeCuts'] = histVtxNContribBeforeCuts.Integral(17,-1)
        histVtxNContribAfterCuts = histRawVtxNContribAfterCuts.to_writable().to_pyroot()
        df.loc[run, 'meanVtxNContribAfterCuts'] = histVtxNContribAfterCuts.GetMean()
        df.loc[run, 'nEventsVtxNContribUnder16AfterCuts'] = histVtxNContribAfterCuts.Integral(1,16)
        df.loc[run, 'nEventsVtxNContribOver15AfterCuts'] = histVtxNContribAfterCuts.Integral(17,-1)

    return df


def get_run_by_run_data_info(dataDir, runList, lumiFilePath="/media/sigurd/T7/analysis/data/LHC23_PbPb_pass5_train590144/mergedAnalysisResults_good.root",
                             mmin=1.81, mmax=1.90, minPt=0., maxPt=12., minY=-0.8, maxY=0.8, text=False, irFilePath='~/cernbox/notebooks/pyD0/interactionRate.txt', fileStructure=FileStructures.RUNDIRS):
    """
    Create a dataframe containing run-by-run quantities for data
    """

    df = pd.read_csv(irFilePath, sep=' ', names=['runNumber', 'ir'])
    df['ir'] = df['ir'].astype(float)
    df['runNumber'] = df['runNumber'].astype(str)
    df = df[df['runNumber'].isin(runList)]
    df = df.set_index('runNumber')

    lumiFile = r.TFile.Open(lumiFilePath)
    lumiTCEHist = lumiFile.Get('eventselection-run3').Get('luminosity').Get('hLumiTCEafterBCcuts')
    lumiTVXHist = lumiFile.Get('eventselection-run3').Get('luminosity').Get('hLumiTVXafterBCcuts')
    lumiZNCHist = lumiFile.Get('eventselection-run3').Get('luminosity').Get('hLumiZNCafterBCcuts')
    counterTCEHist = lumiFile.Get('eventselection-run3').Get('luminosity').Get('hCounterTCEafterBCcuts')
    counterTVXHist = lumiFile.Get('eventselection-run3').Get('luminosity').Get('hCounterTVXafterBCcuts')
    counterZNCHist = lumiFile.Get('eventselection-run3').Get('luminosity').Get('hCounterZNCafterBCcuts')
    colCounterAllHist = lumiFile.Get('eventselection-run3').Get('eventselection').Get('hColCounterAll')
    colCounterTVXHist = lumiFile.Get('eventselection-run3').Get('eventselection').Get('hColCounterTVX')
    colCounterAccHist = lumiFile.Get('eventselection-run3').Get('eventselection').Get('hColCounterAcc')
    for run in runList:
        lumiTCE = lumiTCEHist.GetBinContent(lumiTCEHist.GetXaxis().FindBin(run))
        df.loc[run, 'lumiTCE'] = lumiTCE
        lumiTVX = lumiTVXHist.GetBinContent(lumiTVXHist.GetXaxis().FindBin(run))
        df.loc[run, 'lumiTVX'] = lumiTVX
        lumiZNC = lumiZNCHist.GetBinContent(lumiZNCHist.GetXaxis().FindBin(run))
        df.loc[run, 'lumiZNC'] = lumiZNC
        counterTCE = counterTCEHist.GetBinContent(counterTCEHist.GetXaxis().FindBin(run))
        df.loc[run, 'counterTCE'] = counterTCE
        counterTVX = counterTVXHist.GetBinContent(counterTVXHist.GetXaxis().FindBin(run))
        df.loc[run, 'counterTVX'] = counterTVX
        counterZNC = counterZNCHist.GetBinContent(counterZNCHist.GetXaxis().FindBin(run))
        df.loc[run, 'counterZNC'] = counterZNC
        colCounterAll = colCounterAllHist.GetBinContent(colCounterAllHist.GetXaxis().FindBin(run))
        df.loc[run, 'nEventsColCounterAll'] = colCounterAll
        colCounterTVX = colCounterTVXHist.GetBinContent(colCounterTVXHist.GetXaxis().FindBin(run))
        df.loc[run, 'nEventsColCounterTVX'] = colCounterTVX
        colCounterAcc = colCounterAccHist.GetBinContent(colCounterAccHist.GetXaxis().FindBin(run))
        df.loc[run, 'nEventsColCounterAcc'] = colCounterAcc

    for irun, run in enumerate(runList):
        print(f"Processing run {run} ({irun+1}/{len(runList)})...            ", end='\r')
        if fileStructure is FileStructures.FLAT:
            dataFilepath = f"{dataDir}/AnalysisResults_run{run}.root"
        else:
            dataFilepath = f"{dataDir}/{run}/AnalysisResults.root"

        # Get event info
        with uproot.open(dataFilepath) as dataTmpFile:
            tmpDir = dataTmpFile["analysis-event-selection/output;1"]
        for i, item in enumerate(tmpDir):
            # TODO: Which cuts should be applied when counting these candidates?
            if item.member("fName") == "Event_AfterCuts":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'VtxNContribReal':
                        histRawVtxNContribReal = tmpDir[i][ii]
                    elif iitem.member("fName") == 'VtxNContrib':
                        histRawVtxNContrib = tmpDir[i][ii]
                    elif iitem.member("fName") == 'MyAmpFT0A_AmpFT0CHisto':
                        histRawFT0 = tmpDir[i][ii]
                    elif iitem.member("fName") == 'IsITSUPCMode':
                        histRawIsITSUPCMode = tmpDir[i][ii]
                    elif iitem.member("fName") == 'MyIsITSUPCModeVtxNContribHisto':
                        histRawIsITSUPCModeVtxNContrib = tmpDir[i][ii]

        histVtxNContribReal = histRawVtxNContribReal.to_writable().to_pyroot()
        df.loc[run, 'meanVtxNContribReal'] = histVtxNContribReal.GetMean()
        df.loc[run, 'meanVtxNContribRealError'] = histVtxNContribReal.GetMeanError()
        df.loc[run, 'nEventsVtxNContribRealUnder17'] = histVtxNContribReal.Integral(1,17)
        df.loc[run, 'nEventsVtxNContribRealOver16'] = histVtxNContribReal.Integral(18,-1)
        # Count the number of events in this run
        df.loc[run, 'nEventsAfterCuts'] = histVtxNContribReal.GetEntries()
        histVtxNContrib = histRawVtxNContrib.to_writable().to_pyroot()
        df.loc[run, 'meanVtxNContrib'] = histVtxNContrib.GetMean()
        df.loc[run, 'meanVtxNContribError'] = histVtxNContrib.GetMeanError()
        df.loc[run, 'nEventsVtxNContribUnder16'] = histVtxNContrib.Integral(1,16)
        df.loc[run, 'nEventsVtxNContribOver16'] = histVtxNContrib.Integral(18,-1)
        # Count the number of events in specific ranges of FT0A or FT0C amplitudes
        histFT0 = histRawFT0.to_writable().to_pyroot()
        df.loc[run, 'nEventsFT0AAbove100'] = histFT0.Integral(101, -1, 1, -1)
        df.loc[run, 'nEventsFT0CAbove50'] = histFT0.Integral(1, -1, 51, -1)
        # Count the number of events reconstructed with UPC mode
        histIsITSUPCMode = histRawIsITSUPCMode.to_writable().to_pyroot()
        if histIsITSUPCMode.Integral() < histIsITSUPCMode.GetEntries():
            print(f"WARNING: Run {run} IsITSUPCMode has {histIsITSUPCMode.GetEntries()} entries but the integral is {histIsITSUPCMode.Integral()}!")
            print(f"Bin 1 content = {histIsITSUPCMode.GetBinContent(1)}")
            print(f"Bin 2 content = {histIsITSUPCMode.GetBinContent(2)}")
        df.loc[run, 'nEventsAfterCutsITSUPCModeFalse'] = histIsITSUPCMode.GetBinContent(1)
        df.loc[run, 'nEventsAfterCutsITSUPCModeTrue'] = histIsITSUPCMode.GetBinContent(2)

        try:
            histIsITSUPCModeVtxNContrib = histRawIsITSUPCModeVtxNContrib.to_writable().to_pyroot()
            df.loc[run, 'nEventsVtxNContribUnder16ITSUPCModeFalse'] = histIsITSUPCModeVtxNContrib.Integral(1,16,1,1)
            df.loc[run, 'nEventsVtxNContribOver16ITSUPCModeFalse'] = histIsITSUPCModeVtxNContrib.Integral(18,-1,1,1)
            df.loc[run, 'nEventsVtxNContribUnder16ITSUPCModeTrue'] = histIsITSUPCModeVtxNContrib.Integral(1,16,2,2)
            df.loc[run, 'nEventsVtxNContribOver16ITSUPCModeTrue'] = histIsITSUPCModeVtxNContrib.Integral(18,-1,2,2)
        except:
            print("WARNING: MyIsITSUPCModeVtxNContribHisto was not found in file!")

        # Get pair info
        with uproot.open(dataFilepath) as dataTmpFile:
            tmpDir = dataTmpFile["analysis-asymmetric-pairing/output;1"]
        for i, item in enumerate(tmpDir):
            # TODO: Which cuts should be applied when counting these candidates?
            if item.member("fName") == "PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepLxyCosPointingAngleCut":
                for ii, iitem in enumerate(tmpDir[i]):
                    if iitem.member("fName") == 'MyMassPtYHisto':
                        dataHistRaw = tmpDir[i][ii]

        dataHist = dataHistRaw.to_hist()
        dataNumCandidates = dataHist[complex(0, mmin):complex(0, mmax), complex(0, minPt):complex(0, maxPt), complex(0, minY):complex(0, maxY)].sum().value
        df.loc[run, 'dataNumCandidates'] = dataNumCandidates

    print("Complete!                        ")

    return df

def run_by_run_tof_matching_efficiency(runListFull, runListNo, runListYes, runListExclude, directory, dataOrMc='mc', showPlot=True, firstPtBin=0, lastPtBin=-1, firstEtaBin=0, lastEtaBin=-1, sortBy='yesno'):
    """
    Get <hasTOF> as a function of run number. Runs are placed in two categories: 'no' and 'yes', which may for example represent UPC settings, or CBT runlist.
    The categories only affect visualization and sorting along the x-axis.
    """
    if dataOrMc not in {'mc', 'data'}:
        raise Exception("Argument dataOrMc should be 'data' or 'mc'!")
    if sortBy == 'yesno':
        # Sort runs based on yes/no 
        runListFull = sorted(runListFull, key=lambda run: 1*int(run) if run in runListNo else 10*int(run) if run in runListYes else 100*int(run))
    elif sortBy == 'ir':
        # Get IR per run
        runIrDict = {}
        with open('/home/sigurd/cernbox/singlegap/LHC23_PbPb_pass4_train355821/runInfo.txt', mode='r') as file:
            next(file) # skip header
            for line in file:
                key, value = line.strip().split()
                runIrDict[key] = float(value)
        runListFull = sorted(runListFull, key=lambda run: runIrDict[run])

    print(f"Processing {len(runListFull)} runs ({dataOrMc})...")
    listMeanHasTOF = []
    listMeanHasTOFErrors = []
    listColors = []
    listTextColors = []
    dictColors = {'mcno': 'xkcd:pale lime green', 'datano': 'xkcd:pale cyan', 'mcyes': 'xkcd:pale olive green', 'datayes': 'xkcd:pastel blue', 'mcpartial': 'lightgray', 'datapartial': 'gray'}
    dictTextColors = {'mcno': 'xkcd:frog green', 'datano': 'xkcd:bright sky blue', 'mcyes': 'xkcd:forrest green', 'datayes': 'xkcd:cerulean', 'mcpartial': 'lightgray', 'datapartial': 'gray'}
    # Open a test file to initialize the merged histograms
    if dataOrMc == 'mc':
        testFile = r.TFile.Open(f"{directory}/AnalysisResults_run{runListFull[0]}.root")
        profileHasTOFMergedRunsNotExcludedNo = testFile.Get("analysis-track-selection/output").FindObject("AssocsBarrel_BeforeCuts").FindObject("MyTrackHasTOFMap")
        profileHasTOFMergedAllRunsNo = testFile.Get("analysis-track-selection/output").FindObject("AssocsBarrel_BeforeCuts").FindObject("MyTrackHasTOFMap")
        profileHasTOFMergedRunsNotExcludedYes = testFile.Get("analysis-track-selection/output").FindObject("AssocsBarrel_BeforeCuts").FindObject("MyTrackHasTOFMap")
        profileHasTOFMergedAllRunsYes = testFile.Get("analysis-track-selection/output").FindObject("AssocsBarrel_BeforeCuts").FindObject("MyTrackHasTOFMap")
    else:
        testFile = r.TFile.Open(f"{directory}/{runListFull[0]}/AnalysisResults.root")
        profileHasTOFMergedRunsNotExcludedNo = testFile.Get("analysis-track-selection/output").FindObject("TrackBarrel_MyStandardPrimaryTrackDCACut").FindObject("MyTrackHasTOFMap")
        profileHasTOFMergedAllRunsNo = testFile.Get("analysis-track-selection/output").FindObject("TrackBarrel_MyStandardPrimaryTrackDCACut").FindObject("MyTrackHasTOFMap")
        profileHasTOFMergedRunsNotExcludedYes = testFile.Get("analysis-track-selection/output").FindObject("TrackBarrel_MyStandardPrimaryTrackDCACut").FindObject("MyTrackHasTOFMap")
        profileHasTOFMergedAllRunsYes = testFile.Get("analysis-track-selection/output").FindObject("TrackBarrel_MyStandardPrimaryTrackDCACut").FindObject("MyTrackHasTOFMap")

    profileHasTOFMergedRunsNotExcludedNo = profileHasTOFMergedRunsNotExcludedNo.Project3DProfile('xy') # pT, eta
    profileHasTOFMergedRunsNotExcludedNo = profileHasTOFMergedRunsNotExcludedNo.ProfileY('profileHasTOFMergedRunsNotExcludedNo', firstEtaBin, lastEtaBin) # pT
    profileHasTOFMergedRunsNotExcludedNo.Reset()
    profileHasTOFMergedRunsNotExcludedNo.SetName("profileHasTOFMergedRunsNotExcludedNo")
    profileHasTOFMergedRunsNotExcludedNo.SetTitle("Runs with TOF and 'no'")
    profileHasTOFMergedAllRunsNo = profileHasTOFMergedAllRunsNo.Project3DProfile('xy') # pT, eta
    profileHasTOFMergedAllRunsNo = profileHasTOFMergedAllRunsNo.ProfileY('profileHasTOFMergedAllRunsNo', firstEtaBin, lastEtaBin) # pT
    profileHasTOFMergedAllRunsNo.Reset()
    profileHasTOFMergedAllRunsNo.SetName("profileHasTOFMergedAllRunsNo")
    profileHasTOFMergedAllRunsNo.SetTitle("All runs 'no'")
    profileHasTOFMergedRunsNotExcludedYes = profileHasTOFMergedRunsNotExcludedYes.Project3DProfile('xy') # pT, eta
    profileHasTOFMergedRunsNotExcludedYes = profileHasTOFMergedRunsNotExcludedYes.ProfileY('profileHasTOFMergedRunsNotExcludedYes', firstEtaBin, lastEtaBin) # pT
    profileHasTOFMergedRunsNotExcludedYes.Reset()
    profileHasTOFMergedRunsNotExcludedYes.SetName("profileHasTOFMergedRunsNotExcludedYes")
    profileHasTOFMergedRunsNotExcludedYes.SetTitle("Runs with TOF and 'yes'")
    profileHasTOFMergedAllRunsYes = profileHasTOFMergedAllRunsYes.Project3DProfile('xy') # pT, eta
    profileHasTOFMergedAllRunsYes = profileHasTOFMergedAllRunsYes.ProfileY('profileHasTOFMergedAllRunsYes', firstEtaBin, lastEtaBin) # pT
    profileHasTOFMergedAllRunsYes.Reset()
    profileHasTOFMergedAllRunsYes.SetName("profileHasTOFMergedAllRunsYes")
    profileHasTOFMergedAllRunsYes.SetTitle("All runs 'yes'")
    # Check the merged histograms
    """
    cTest = r.TCanvas("cTest", "cTest", 1400, 500)
    cTest.Divide(3,1)
    cTest.cd(1)
    profileHasTOFMergedRunsNotExcludedNo.Draw()
    cTest.cd(2)
    profileHasTOFMergedAllRunsNo.Draw()
    cTest.cd(3)
    profileHasTOFMergedAllRunsYes.Draw()
    cTest.Draw()
    """

    for i, run in enumerate(runListFull):
        if dataOrMc == 'mc':
            filepath = f"{directory}/AnalysisResults_run{run}.root"
        else:
            filepath = f"{directory}/{run}/AnalysisResults.root"

        tmpFile = r.TFile.Open(filepath)
        if not tmpFile or tmpFile.IsZombie():
            print(f"Unable to open file for run {run}!")
            continue

        # Sanity check
        if dataOrMc == 'mc':
            tmpHistRunNumberCheck = tmpFile.Get("lumi-task/hCounterTCE")
        else:
            tmpHistRunNumberCheck = tmpFile.Get("analysis-event-selection/output").FindObject("Event_BeforeCuts").FindObject("VtxZ_Run")
        if (tmpHistRunNumberCheck.GetNbinsX() != 1):
            print(f"File for run {run} contains data from more than one run number!")
            continue
        if (tmpHistRunNumberCheck.GetXaxis().GetLabels().At(0).GetName() != run):
            print(f"File for run {run} contains data on the wrong run!")

        try:
            if dataOrMc == 'mc':
                tmpHist = tmpFile.Get("analysis-track-selection/output").FindObject("AssocsBarrel_BeforeCuts").FindObject("MyTrackHasTOFMap")
            else:
                tmpHist = tmpFile.Get("analysis-track-selection/output").FindObject("TrackBarrel_MyStandardPrimaryTrackDCACut").FindObject("MyTrackHasTOFMap")
        except:
            print(f"Unable to get histograms for run {run}!")
            continue
        minPt = tmpHist.GetXaxis().GetBinLowEdge(firstPtBin)
        maxPt = tmpHist.GetXaxis().GetBinUpEdge(lastPtBin)
        minEta = tmpHist.GetYaxis().GetBinLowEdge(firstEtaBin)
        maxEta = tmpHist.GetYaxis().GetBinUpEdge(lastEtaBin)
        tmpHist = tmpHist.Project3DProfile('xy') # pT, eta
        tmpHist = tmpHist.ProfileY('tmpHist', firstEtaBin, lastEtaBin) # pT
        tmpHist.GetXaxis().SetRange(firstPtBin, lastPtBin)

        listMeanHasTOF.append(tmpHist.GetMean(2))
        listMeanHasTOFErrors.append(tmpHist.GetMeanError(2))
        if run in runListNo:
            listColors.append(dictColors[dataOrMc+'no'])
            if run not in runListExclude:
                profileHasTOFMergedRunsNotExcludedNo.Add(tmpHist)
                listTextColors.append(dictTextColors[dataOrMc+'no'])
            else:
                listTextColors.append('red')
            profileHasTOFMergedAllRunsNo.Add(tmpHist)
        elif run in runListYes:
            listColors.append(dictColors[dataOrMc+'yes'])
            if run not in runListExclude:
                profileHasTOFMergedRunsNotExcludedYes.Add(tmpHist)
                listTextColors.append(dictTextColors[dataOrMc+'yes'])
            else:
                listTextColors.append('red')
            profileHasTOFMergedAllRunsYes.Add(tmpHist)
        else:
            listColors.append(dictColors[dataOrMc+'partial'])
            if run not in runListExclude:
                listTextColors.append(dictTextColors[dataOrMc+'yes'])
            else:
                listTextColors.append('red')
        tmpFile.Close()

    # Extract the <hasTOF> averaged over the different sets of runs
    profileHasTOFMergedAllRunsNo.GetXaxis().SetRange(firstPtBin, lastPtBin)
    meanHasTOFMergedAllRunsNo = profileHasTOFMergedAllRunsNo.GetMean(2)
    meanErrorHasTOFMergedAllRunsNo = profileHasTOFMergedAllRunsNo.GetMeanError(2)
    print(f"All runs 'no': <hasTOF> = {meanHasTOFMergedAllRunsNo:.4f} +- {meanErrorHasTOFMergedAllRunsNo:.4f}")
    profileHasTOFMergedRunsNotExcludedNo.GetXaxis().SetRange(firstPtBin, lastPtBin)
    meanHasTOFMergedRunsNotExcludedNo = profileHasTOFMergedRunsNotExcludedNo.GetMean(2)
    meanErrorHasTOFMergedRunsNotExcludedNo = profileHasTOFMergedRunsNotExcludedNo.GetMeanError(2)
    if runListExclude:
        print(f"Runs not excluded and 'no: <hasTOF> = {meanHasTOFMergedRunsNotExcludedNo:.4f} +- {meanErrorHasTOFMergedRunsNotExcludedNo:.4f}")
    profileHasTOFMergedAllRunsYes.GetXaxis().SetRange(firstPtBin, lastPtBin)
    meanHasTOFMergedAllRunsYes = profileHasTOFMergedAllRunsYes.GetMean(2)
    meanErrorHasTOFMergedAllRunsYes = profileHasTOFMergedAllRunsYes.GetMeanError(2)
    print(f"All runs 'yes': <hasTOF> = {meanHasTOFMergedAllRunsYes:.4f} +- {meanErrorHasTOFMergedAllRunsYes:.4f}")
    profileHasTOFMergedRunsNotExcludedYes.GetXaxis().SetRange(firstPtBin, lastPtBin)
    meanHasTOFMergedRunsNotExcludedYes = profileHasTOFMergedRunsNotExcludedYes.GetMean(2)
    meanErrorHasTOFMergedRunsNotExcludedYes = profileHasTOFMergedRunsNotExcludedYes.GetMeanError(2)
    if runListExclude:
        print(f"Runs not excluded and 'yes': <hasTOF> = {meanHasTOFMergedRunsNotExcludedYes:.4f} +- {meanErrorHasTOFMergedRunsNotExcludedYes:.4f}")

    runAxis = np.arange(len(runListFull))
    cutInfoString = ""
    if firstPtBin != -1 or lastPtBin != -1 or firstEtaBin != -1 or lastEtaBin != -1:
        cutInfoString = f". {minPt:.1f} < pT < {maxPt:.1f} GeV/c, {minEta:.1f} < $\eta$ < {maxEta:.1f}"
    if showPlot:
        plt.figure(figsize=(15, 5))
        if dataOrMc == 'mc':
            plt.title(f"MC, reconstructed tracks before cuts")
        else:
            plt.title(f"Data, tracks with |DCAxy| < 0.1 cm, |DCAz| < 0.15 cm")
        plt.bar(runAxis, listMeanHasTOF, yerr=listMeanHasTOFErrors, color=listColors, capsize=3)
        for i, label in enumerate(runListFull):
            plt.text(runAxis[i], -0.005, label, ha='center', va='top', color=listTextColors[i], rotation=90)
        plt.xticks(runAxis, '', rotation=90)
        plt.ylabel('Track <hasTOF>')
        plt.hlines(meanHasTOFMergedAllRunsNo, -0.5, len(runListNo)-0.5, alpha=0.5, color=dictTextColors[dataOrMc+'no'], linestyle='dashed')
        plt.hlines(meanHasTOFMergedRunsNotExcludedNo, -0.5, len(runListNo)-0.5, alpha=0.5, color=dictTextColors[dataOrMc+'no'])
        plt.hlines(meanHasTOFMergedAllRunsYes, len(runListNo)-0.5, len(runListNo)+len(runListYes)-0.5, alpha=0.5, color=dictTextColors[dataOrMc+'yes'], linestyle='dashed')
        plt.hlines(meanHasTOFMergedRunsNotExcludedYes, len(runListNo)-0.5, len(runListNo)+len(runListYes)-0.5, alpha=0.5, color=dictTextColors[dataOrMc+'yes'])
        plt.tight_layout()
        plt.show()
    testFile.Close()
    return {'runList': runListFull, 'runAxis': runAxis, 'values': listMeanHasTOF, 'errors': listMeanHasTOFErrors, 'colors': listColors, 'textColors': listTextColors, 'dictColors': dictColors, 'minPt': minPt, 'maxPt': maxPt, 'minEta': minEta, 'maxEta': maxEta}

class OccupancyStudy():
    def __init__(self, fileDirectory=None, filePath=None,
                 runList=DEFAULT_RUNLIST_PASS4.copy(), contribBins=[0, 2000, 4000, 20000], timeBins=[-100, -40, 40, 100], minPt=0, maxPt=12):
        r.gStyle.SetPalette(r.kPastel)
        self.massRange = [0., 4.]
        tmpFile = r.TFile.Open(filePath)
        self.histOrig = tmpFile.Get("analysis-asymmetric-pairing/output").FindObject("PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepTauxyzprojCut").FindObject("MyD0OccupHisto")
        # self.histOrig = tmpFile.Get("analysis-same-event-pairing/output").FindObject("PairsBarrelSEPM_muonCandidateCut").FindObject("MyD0OccupHisto")
        self.histEvents = tmpFile.Get("analysis-event-selection/output").FindObject("Event_AfterCuts").FindObject("MyOccupHisto")
        self.histOrig.GetAxis(1).SetRangeUser(minPt, maxPt)
        print(f"Projecting pT from bin {self.histOrig.GetAxis(1).FindBin(minPt)} ({self.histOrig.GetAxis(1).GetBinLowEdge(self.histOrig.GetAxis(1).FindBin(minPt))} GeV) to {self.histOrig.GetAxis(1).FindBin(maxPt)} ({self.histOrig.GetAxis(1).GetBinUpEdge(self.histOrig.GetAxis(1).FindBin(maxPt))} GeV)")
        self.hist = self.histOrig.Projection(0, 2, 3)
        # Make slices of the occupancy variable
        self.slices = {}
        self.fitResults = {}
        self.textBoxes = {}
        massBins = [self.hist.GetXaxis().GetBinLowEdge(1+i) for i in range(self.hist.GetXaxis().GetNbins())]
        massBins.append(self.hist.GetXaxis().GetBinLowEdge(self.hist.GetXaxis().GetNbins()))
        massBins = np.array(massBins, dtype='d')
        self.histRebinned = r.TH3D(f"{self.hist.GetName()}_rebinned", f"{self.hist.GetTitle()}",
                                   self.hist.GetXaxis().GetNbins(), massBins,
                                   len(contribBins) - 1, np.array(contribBins, dtype='d'),
                                   len(timeBins) - 1, np.array(timeBins, dtype='d'))
        self.histRebinned.SetDirectory(0)
        for ix in range(1, self.hist.GetXaxis().GetNbins() + 1):
            x = self.hist.GetXaxis().GetBinCenter(ix)
            for iy in range(1, self.hist.GetYaxis().GetNbins() + 1):
                y = self.hist.GetYaxis().GetBinCenter(iy)
                for iz in range(1, self.hist.GetZaxis().GetNbins() + 1):
                    z = self.hist.GetZaxis().GetBinCenter(iz)
                    val = self.hist.GetBinContent(ix, iy, iz)
                    err = self.hist.GetBinError(ix, iy, iz)
                    if val == 0 and err == 0:
                        continue
                    # Find target bin in rebinned histogram
                    ix_new = self.histRebinned.GetXaxis().FindBin(x)
                    iy_new = self.histRebinned.GetYaxis().FindBin(y)
                    iz_new = self.histRebinned.GetZaxis().FindBin(z)
                    # Current values in the target bin
                    old_val = self.histRebinned.GetBinContent(ix_new, iy_new, iz_new)
                    old_err = self.histRebinned.GetBinError(ix_new, iy_new, iz_new)
                    # New accumulated content and error
                    new_val = old_val + val
                    new_err = (old_err**2 + err**2)**0.5  # sum errors in quadrature
                    self.histRebinned.SetBinContent(ix_new, iy_new, iz_new, new_val)
                    self.histRebinned.SetBinError(ix_new, iy_new, iz_new, new_err)
        self.histEventsRebinned = r.TH2D(f"{self.histEvents.GetName()}_rebinned", f"{self.histEvents.GetTitle()}",
                                   len(contribBins) - 1, np.array(contribBins, dtype='d'),
                                   len(timeBins) - 1, np.array(timeBins, dtype='d'))
        self.histEventsRebinned.SetDirectory(0)
        self.histEventsRebinned.GetXaxis().SetTitle(self.histEvents.GetXaxis().GetTitle())
        self.histEventsRebinned.GetYaxis().SetTitle(self.histEvents.GetYaxis().GetTitle())
        for ix in range(1, self.histEvents.GetXaxis().GetNbins() + 1):
            x = self.histEvents.GetXaxis().GetBinCenter(ix)
            for iy in range(1, self.histEvents.GetYaxis().GetNbins() + 1):
                y = self.histEvents.GetYaxis().GetBinCenter(iy)
                val = self.histEvents.GetBinContent(ix, iy)
                err = self.histEvents.GetBinError(ix, iy)
                if val == 0 and err == 0:
                    continue
                # Find target bin in rebinned histogram
                ix_new = self.histEventsRebinned.GetXaxis().FindBin(x)
                iy_new = self.histEventsRebinned.GetYaxis().FindBin(y)
                # Current values in the target bin
                old_val = self.histEventsRebinned.GetBinContent(ix_new, iy_new)
                old_err = self.histEventsRebinned.GetBinError(ix_new, iy_new)
                # New accumulated content and error
                new_val = old_val + val
                new_err = (old_err**2 + err**2)**0.5  # sum errors in quadrature
                self.histEventsRebinned.SetBinContent(ix_new, iy_new, new_val)
                self.histEventsRebinned.SetBinError(ix_new, iy_new, new_err)

        for i in range(1, self.histRebinned.GetYaxis().GetNbins() + 1):
            for j in range(1, self.histRebinned.GetZaxis().GetNbins() + 1):
                tmpHist = self.histRebinned.ProjectionX(f"{i}_{j}", iymin=i, iymax=i, izmin=j, izmax=j)
                tmpHist.SetTitle(f"{self.histRebinned.GetYaxis().GetBinLowEdge(i)}<NTPCcontribLongA<{self.histRebinned.GetYaxis().GetBinUpEdge(i)}, {self.histRebinned.GetZaxis().GetBinLowEdge(j)}<NTPCmedianTimeLongA<{self.histRebinned.GetZaxis().GetBinUpEdge(j)}")
                h = tmpHist.Clone(f"{self.histRebinned.GetName()}_{i}_{j}")
                h.SetDirectory(0)
                self.slices[(i,j)] = h

    def draw_inv_mass(self, nW=None, nH=None):
        if nW is None and nH is None:
            nW = self.histRebinned.GetYaxis().GetNbins()
            nH = self.histRebinned.GetZaxis().GetNbins()
            if nW == 1:
                nW = nH
                nH = 1
        self.cGridInvM = r.TCanvas('cGridInvM', 'cGridInvM', 500*nW, 400*nH)
        self.cGridInvM.Divide(nW, nH, 1e-10, 1e-10)
        count = 0
        for j in range(1, self.histRebinned.GetZaxis().GetNbins() + 1):
            for i in range(1, self.histRebinned.GetYaxis().GetNbins() + 1):
                count += 1
                self.cGridInvM.cd(count)
                self.slices[(i,j)].Draw()
                if (i,j) in self.fitResults:
                    self.textBoxes[(i,j)] = r.TPaveText(0.5, 0.25, 0.9, 0.65, "NDC")
                    self.textBoxes[(i,j)].SetName(f"textbox_{i}_{j}")
                    self.textBoxes[(i,j)].SetFillColor(0)
                    self.textBoxes[(i,j)].SetFillStyle(0)
                    self.textBoxes[(i,j)].SetBorderSize(0)
                    self.textBoxes[(i,j)].SetTextAlign(12)
                    self.textBoxes[(i,j)].SetTextFont(42)
                    self.textBoxes[(i,j)].SetTextSize(0.04)
                    self.textBoxes[(i,j)].AddText(f"#mu = {self.fitResults[(i,j)].fitMu:.3f}")
                    self.textBoxes[(i,j)].AddText(f"#sigma = {self.fitResults[(i,j)].fitSigma:.3f}")
                    self.textBoxes[(i,j)].AddText(f"S = {self.fitResults[(i,j)].nSignal:.0f} #pm {self.fitResults[(i,j)].nSignal*self.fitResults[(i,j)].relativeStatError:.0f}")
                    self.textBoxes[(i,j)].AddText(f"S/B (3#sigma)= {self.fitResults[(i,j)].signalToBackground:.3f}")
                    self.textBoxes[(i,j)].AddText(f"S/#sqrt{{S+B}} = {self.fitResults[(i,j)].signalSignificance:.3f}")
                    self.textBoxes[(i,j)].AddText(f"S/Evt = {self.fitResults[(i,j)].yieldPerEvent:.3E} #pm {self.fitResults[(i,j)].yieldPerEventError:.3E}")
                    self.textBoxes[(i,j)].AddText(f"#chi^{{2}}/ndf = {self.fitResults[(i,j)].fitChi2Ndf:.3f}")
                    self.textBoxes[(i,j)].Draw()
                self.slices[(i,j)].SetAxisRange(0.0,
                                                self.slices[(i,j)].GetBinContent(self.slices[(i,j)].FindBin(self.fitResults[(i,j)].fitMu))*1.1,
                                                'Y')
        self.cGridInvM.Draw()
        self.cGridInvM.Modified()
        self.cGridInvM.Update()

    def calculate_yield_per_event(self, indices):
        self.fitResults[indices].yieldPerEvent = self.fitResults[indices].nSignal / self.histEventsRebinned.GetBinContent(indices[0], indices[1])
        self.fitResults[indices].yieldPerEventError = self.fitResults[indices].yieldPerEvent * np.sqrt((self.fitResults[indices].relativeStatError)**2 + (self.histEventsRebinned.GetBinError(indices[0], indices[1]) / self.histEventsRebinned.GetBinContent(indices[0], indices[1]))**2)

    def create_yield_per_event_histogram(self):
        self.histYieldPerEvent = self.histEventsRebinned.Clone("histYieldPerEvent")
        self.histYieldPerEvent.Reset()
        self.histYieldPerEvent.SetTitle("Yield per event")
        self.histYieldPerEvent.GetXaxis().SetTitle(self.histEvents.GetXaxis().GetTitle())
        self.histYieldPerEvent.GetYaxis().SetTitle(self.histEvents.GetYaxis().GetTitle())
        for i in range(1, self.histYieldPerEvent.GetXaxis().GetNbins() + 1):
            for j in range(1, self.histYieldPerEvent.GetYaxis().GetNbins() + 1):
                self.histYieldPerEvent.SetBinContent(i, j, self.fitResults[(i, j)].yieldPerEvent)
                self.histYieldPerEvent.SetBinError(i, j, self.fitResults[(i, j)].yieldPerEventError)

    def draw_yield_per_event(self, width=1200, height=800):
        if not hasattr(self, 'histYieldPerEvent'):
            self.create_yield_per_event_histogram()
        self.cYieldPerEvent = r.TCanvas('cYieldPerEvent', 'cYieldPerEvent', width, height)
        self.cYieldPerEvent.cd()
        self.histYieldPerEvent.Draw('colz texte')
        self.histYieldPerEvent.SetStats(0)
        self.cYieldPerEvent.Draw()

    def draw_yield_per_event_projections(self):
        if not hasattr(self, 'histYieldPerEvent'):
            self.create_yield_per_event_histogram()
        self.histYieldPerEventProjPileup = self.histYieldPerEvent.ProjectionX('histYieldPerEventProjPileup', firstybin=0, lastybin=-1)
        self.histYieldPerEventProjTime = self.histYieldPerEvent.ProjectionY('histYieldPerEventProjTime', firstxbin=0, lastxbin=-1)
        self.cProjections = r.TCanvas('cProjections', 'cProjections', 1600, 600)
        self.cProjections.Divide(2,1)
        self.cProjections.cd(1)
        self.histYieldPerEventProjPileup.Draw()
        self.histYieldPerEventProjPileup.SetStats(0)
        self.cProjections.cd(2)
        self.histYieldPerEventProjTime.Draw()
        self.histYieldPerEventProjTime.SetStats(0)
        self.cProjections.Draw()

    def fit_inv_mass_jpsi(self, histogram, identifier, backgroundFunction, fitRange = [2.5, 3.5], doPrint=True,
                          signalMuRange=None, signalSigmaRange=None,    
                          backgroundInitialParams=None):
        # Create the results object to save info on this fit
        fitResult = FitResult(fitRange[0], fitRange[1], backgroundFunction.__name__)
        if doPrint:
            print(f"====== Fitting {identifier} ({histogram.GetTitle()}) ======")

        backgroundNPar = fit_functions.background_npar[backgroundFunction.__name__]
        if backgroundInitialParams is not None and len(backgroundInitialParams) != backgroundNPar:
            print(f"WARNING: {len(backgroundInitialParams)} initial parameters for combinatorial background specified, but '{backgroundFunction.__name__}' takes {backgroundNPar} parameters!")
        # Get the complete function to be used for the fit
        fitFunc = fit_functions.make_jpsi_fit_model(backgroundFunction, backgroundNPar, fitRange[0], fitRange[1])
        fitFunc.SetName(f"fitFunc_bin_{identifier}")
        fitFunc.SetNpx(1000)
        # Parameters from presentation by Anna Binoy and Amrit Gautam in PAG-UPC Oct 28 2025
        fitFunc.SetParameter(0, 1000) # N
        fitFunc.SetParameter(1, 3.07) # m0
        fitFunc.SetParameter(2, 0.019) # sigma
        fitFunc.FixParameter(3, 1.3009) # alphaL
        fitFunc.FixParameter(4, 5.4258) # nL
        fitFunc.FixParameter(5, 1.8187) # alphaR
        fitFunc.FixParameter(6, 12.3646) # nR
        # Set the background parameters to safe defaults (avoid division by 0)
        for i in range(backgroundNPar):
            fitFunc.SetParameter(7 + i, 1.0 if backgroundInitialParams is None else backgroundInitialParams[i])
        if doPrint:
            print("Initially, the combinatorial background function parameters are:")
            for i in range(backgroundNPar):
                print(f"par[{7 + i}] = {fitFunc.GetParameter(7 + i)}")
        # If specified, set range for the signal width
        if signalSigmaRange is not None:
            fitFunc.SetParLimits(2, signalSigmaRange[0], signalSigmaRange[1])
        # If specified, set range for the signal mean
        if signalMuRange is not None:
            fitFunc.SetParLimits(1, signalMuRange[0], signalMuRange[1])
        fitResult.result = histogram.Fit(fitFunc, "MES" + ("Q" if not doPrint else ""), "", fitRange[0], fitRange[1])
        # Store the functions for plotting and calculations
        fitFuncParams = [fitFunc.GetParameters()[i] for i in range(fitFunc.GetNpar())]
        fitFuncParErrors = [fitFunc.GetParErrors()[i] for i in range(fitFunc.GetNpar())]
        if doPrint:
            print(f"fitFuncParams = {[str(par)+'+-'+str(err) for (par, err) in zip(fitFuncParams, fitFuncParErrors)]}")
        signalFunc = r.TF1(f"signalFunc_bin{bin}", fit_functions.dscb, self.massRange[0], self.massRange[1], 7)
        signalFunc.SetParameters(np.array(fitFuncParams[0:7], dtype='d'))
        signalFunc.SetParErrors(np.array(fitFuncParErrors[0:7], dtype='d'))
        backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", backgroundFunction, self.massRange[0], self.massRange[1], backgroundNPar)
        backgroundFunc.SetParameters(np.array(fitFuncParams[-backgroundNPar:], dtype='d'))
        backgroundFunc.SetParErrors(np.array(fitFuncParErrors[-backgroundNPar:], dtype='d'))
        backgroundFunc.SetLineColor(r.kBlue)
        try:
            fitResult.fitChi2Ndf = fitFunc.GetChisquare() / fitFunc.GetNDF()
        except:
            print("WARNING: Did not manage to calculate the chi2/ndf for the fit function!")
            fitResult.fitChi2Ndf = 0
        fitResult.fitMu = fitFunc.GetParameter('m0')
        fitResult.fitSigma = fitFunc.GetParameter('sigma')
        fitResult.nSignal = signalFunc.Integral(fitResult.fitMu - 3*fitResult.fitSigma, fitResult.fitMu + 3*fitResult.fitSigma) / histogram.GetBinWidth(1)
        fitResult.nCombBackground = backgroundFunc.Integral(fitResult.fitMu - 3*fitResult.fitSigma, fitResult.fitMu + 3*fitResult.fitSigma) / histogram.GetBinWidth(1)
        if doPrint:
            print(f"{identifier} nCombBackground = {fitResult.nCombBackground} was calculated by integration from {fitResult.fitMu - 3*fitResult.fitSigma} to {fitResult.fitMu + 3*fitResult.fitSigma}")
        fitResult.signalToBackground = fitResult.nSignal / (fitResult.nCombBackground)
        fitResult.signalSignificance = fitResult.nSignal / np.sqrt(fitResult.nSignal + fitResult.nCombBackground)
        fitResult.relativeStatError = 1. / fitResult.signalSignificance
        if doPrint:
            print(f"{identifier} signal, significance, statistical error, and S/B were calculated by integrating the signal and background components from {fitResult.fitMu - 3*fitResult.fitSigma} to {fitResult.fitMu + 3*fitResult.fitSigma}, yielding nSig={fitResult.nSignal}, nComb={fitResult.nCombBackground}")
        return fitResult
        # self.fitResults[indices] = fitResult
        # self.calculate_yield_per_event(indices)

    def fit_inv_mass_D0(self, indices, backgroundFunction, fitRange = [1.64, 2.08], doPrint=True,
                        signalMuRange=None, signalSigmaRange=None,    
                        backgroundInitialParams=None,
                        simpleFitInitialParams=[100, 1.85, 0.015, 100, 0],
                        simpleFitRange=[1.75, 1.98],
                        shadowRange=[1.8, 1.9]):
        # Create the results object to save info on this fit
        fitResult = FitResult(fitRange[0], fitRange[1], backgroundFunction.__name__, 0)
        if doPrint:
            print(f"====== Fitting bin {indices} ({self.slices[indices].GetTitle()}) ======")

        backgroundNPar = fit_functions.background_npar[backgroundFunction.__name__]
        if backgroundInitialParams is not None and len(backgroundInitialParams) != backgroundNPar:
            print(f"WARNING: {len(backgroundInitialParams)} initial parameters for combinatorial background specified, but '{backgroundFunction.__name__}' takes {backgroundNPar} parameters!")
        # Get the complete function to be used for the fit
        fitFunc = fit_functions.make_fit_model(backgroundFunction, backgroundNPar, fitRange[0], fitRange[1])
        fitFunc.SetName(f"fitFunc_bin_{indices[0]}_{indices[1]}")
        # Fix the parameters of the reflected components
        # TODO: These parameters should have some errors associated with them, coming from to the fit to MC
        fitFunc.FixParameter(3, 0) # Ratio refl/S
        fitFunc.FixParameter(4, 0) # Relative normalization of the two Gaussians
        fitFunc.FixParameter(5, 0)
        fitFunc.FixParameter(6, 1)
        fitFunc.FixParameter(7, 0)
        fitFunc.FixParameter(8, 1)
        # Use a simple linear background to get initial parameters for the signal function
        simpleFitFunc = r.TF1(f"simpleFitFunc_bin_{indices[0]}_{indices[1]}", fit_functions.simple_fit_model, 1.7, 2.03, npar=5, ndim=1)
        simpleFitFunc.SetParameters(*simpleFitInitialParams)
        # If specified, set range for the signal width
        if signalSigmaRange is not None:
            simpleFitFunc.SetParLimits(2, signalSigmaRange[0], signalSigmaRange[1])
        # If specified, set range for the signal mean
        if signalMuRange is not None:
            simpleFitFunc.SetParLimits(1, signalMuRange[0], signalMuRange[1])
        self.slices[indices].Fit(simpleFitFunc, "ME0" + ("Q" if not doPrint else ""), "", simpleFitRange[0], simpleFitRange[1])
        fitFunc.SetParameter(0, simpleFitFunc.GetParameter(0)) # Signal yield
        fitFunc.SetParameter(1, simpleFitFunc.GetParameter(1)) # Mean
        fitFunc.SetParameter(2, simpleFitFunc.GetParameter(2)) # Sigma
        # Set the background parameters to safe defaults (avoid division by 0)
        for i in range(backgroundNPar):
            fitFunc.SetParameter(3 + 6 + i, 1.0 if backgroundInitialParams is None else backgroundInitialParams[i])
        if doPrint:
            print("Initially, the combinatorial background function parameters are:")
            for i in range(backgroundNPar):
                print(f"par[{3 + 6 + i}] = {fitFunc.GetParameter(3 + 6 + i)}")
        # Obtain a shadowed histogram with signal region removed, and do an initial fit to the background
        sliceShadowed = self.slices[indices].Clone(f"invMassShadowed_bin_{indices[0]}_{indices[1]}")
        for i in range(1, sliceShadowed.GetNbinsX() + 1):
            binCenter = sliceShadowed.GetBinCenter(i)
            if binCenter > shadowRange[0] and binCenter < shadowRange[1]:
                sliceShadowed.SetBinContent(i, 0.0)
                sliceShadowed.SetBinError(i, 0.0)
        # Fix the signal yield, mean and sigma to the previously obtained values
        fitFunc.FixParameter(0, simpleFitFunc.GetParameter(0)) # Signal yield
        fitFunc.FixParameter(1, simpleFitFunc.GetParameter(1)) # Mean
        fitFunc.FixParameter(2, simpleFitFunc.GetParameter(2)) # Sigma
        sliceShadowed.Fit(fitFunc, "ME" + ("Q" if not doPrint else ""), "", fitRange[0], fitRange[1])
        if doPrint:
            print("After fit to shadowed histogram, the combinatorial background function parameters are:")
            for i in range(backgroundNPar):
                print(f"par[{3 + 6 + i}] = {fitFunc.GetParameter(3 + 6 + i)}")
        # The parameters for the combinatorial background should now have reasonable values. Now release the signal function parameters for the final fit
        fitFunc.ReleaseParameter(0) # Signal yield
        fitFunc.ReleaseParameter(1) # Mean
        fitFunc.ReleaseParameter(2) # Sigma
        # If specified, set range for the signal width
        if signalSigmaRange is not None:
            fitFunc.SetParLimits(2, signalSigmaRange[0], signalSigmaRange[1])
        # If specified, set range for the signal mean
        if signalMuRange is not None:
            fitFunc.SetParLimits(1, signalMuRange[0], signalMuRange[1])
        fitResult.result = self.slices[indices].Fit(fitFunc, "MES" + ("Q" if not doPrint else ""), "", fitRange[0], fitRange[1])
        # Store the functions for plotting and calculations
        fitFuncParams = [fitFunc.GetParameters()[i] for i in range(fitFunc.GetNpar())]
        fitFuncParErrors = [fitFunc.GetParErrors()[i] for i in range(fitFunc.GetNpar())]
        nReflErr = utils.propagate_error_product(fitFuncParams[0], fitFuncParams[3], 
                                                 fitFuncParErrors[0], fitFuncParErrors[3],
                                                 fitResult.result.GetCovarianceMatrix()[0,3])
        if doPrint:
            print(f"fitFuncParams = {[str(par)+'+-'+str(err) for (par, err) in zip(fitFuncParams, fitFuncParErrors)]}")
        signalFunc = r.TF1(f"signalFunc_bin{bin}", fit_functions.signal, self.massRange[0], self.massRange[1], 3)
        signalFunc.SetParameters(np.array(fitFuncParams[0:3], dtype='d'))
        signalFunc.SetParErrors(np.array(fitFuncParErrors[0:3], dtype='d'))
        backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", backgroundFunction, self.massRange[0], self.massRange[1], backgroundNPar)
        backgroundFunc.SetParameters(np.array(fitFuncParams[-backgroundNPar:], dtype='d'))
        backgroundFunc.SetParErrors(np.array(fitFuncParErrors[-backgroundNPar:], dtype='d'))
        backgroundFunc.SetLineColor(r.kBlue)
        dataReflFunc = r.TF1(f"dataReflFunc_bin{bin}", fit_functions.reflected_background, self.massRange[0], self.massRange[1], 6)
        dataReflFunc.SetParameters(np.concatenate(([fitFuncParams[0] * fitFuncParams[3]],
                                                                    fitFuncParams[4:9]), dtype='d'))
        dataReflFunc.SetParErrors(np.concatenate(([nReflErr], fitFuncParErrors[4:9]), dtype='d'))
        dataReflFunc.SetLineColor(r.kGreen - 1)
        dataReflFunc.SetLineStyle(r.kDashed)
        totalBackgroundFunc = fit_functions.make_background_model(backgroundFunction, backgroundNPar, self.massRange[0], self.massRange[1])
        totalBackgroundFunc.SetParameters(np.concatenate(([fitFuncParams[0] * fitFuncParams[3]],
                                                                    fitFuncParams[4:]), dtype='d'))
        totalBackgroundFunc.SetParErrors(np.concatenate(([nReflErr], fitFuncParErrors[4:]), dtype='d'))
        totalBackgroundFunc.SetLineColor(r.kGreen)
        totalBackgroundFunc.SetLineStyle(r.kDashed)
        try:
            fitResult.fitChi2Ndf = fitFunc.GetChisquare() / fitFunc.GetNDF()
        except:
            print("WARNING: Did not manage to calculate the chi2/ndf for the fit function!")
            fitResult.fitChi2Ndf = 0
        fitResult.fitMu = fitFunc.GetParameter('Mean')
        fitResult.fitSigma = fitFunc.GetParameter('Sigma')
        fitResult.nSignal = fitFunc.GetParameter(0) / self.slices[indices].GetBinWidth(1)
        fitResult.nCombBackground = backgroundFunc.Integral(fitResult.fitMu - 3*fitResult.fitSigma, fitResult.fitMu + 3*fitResult.fitSigma) / self.slices[indices].GetBinWidth(1)
        if doPrint:
            print(f"bin {indices} nCombBackground = {fitResult.nCombBackground} was calculated by integration from {fitResult.fitMu - 3*fitResult.fitSigma} to {fitResult.fitMu + 3*fitResult.fitSigma}")
        fitResult.nReflBackground = dataReflFunc.GetParameter(0) / self.slices[indices].GetBinWidth(1)
        fitResult.relativeStatError = fitFunc.GetParError(0) / (self.slices[indices].GetBinWidth(1) * fitResult.nSignal)
        signalIntegral3Sigma = signalFunc.Integral(fitResult.fitMu - 3*fitResult.fitSigma, fitResult.fitMu + 3*fitResult.fitSigma) / self.slices[indices].GetBinWidth(1)
        reflIntegral3Sigma = dataReflFunc.Integral(fitResult.fitMu - 3*fitResult.fitSigma, fitResult.fitMu + 3*fitResult.fitSigma) / self.slices[indices].GetBinWidth(1)
        fitResult.signalToBackground = signalIntegral3Sigma / (fitResult.nCombBackground + reflIntegral3Sigma)
        fitResult.signalSignificance = signalIntegral3Sigma / np.sqrt(signalIntegral3Sigma + fitResult.nCombBackground + reflIntegral3Sigma)
        if doPrint:
            print(f"bin {indices} signal significance and S/B were calculated by integrating the signal and background components from {fitResult.fitMu - 3*fitResult.fitSigma} to {fitResult.fitMu + 3*fitResult.fitSigma}, yielding nSig={signalIntegral3Sigma}, nComb={fitResult.nCombBackground}, nReflBackground (3#sigma)={reflIntegral3Sigma}")
        self.fitResults[indices] = fitResult
        self.calculate_yield_per_event(indices)

    def calculate_yield_vs_ir(self, dir, lumiFilePath, interactionRateFile='./interactionRate.txt', excludedRuns=[]):
        # Get lumi hist
        lumiFile = r.TFile.Open(lumiFilePath)
        self.histLumi = lumiFile.Get('eventselection-run3').Get('luminosity').Get('hLumiTCEafterBCcuts')
        # Put interaction rates in a dict
        ir = {}
        with open(interactionRateFile) as f:
            for line in f:
                (key, val) = line.split()
                ir[str(key)] = float(val)
        self.runHists = {}
        self.runFitResults = {}
        listRun = []
        listIr = []
        listYield = []
        listEvents = []
        listEventsSelected = []
        listYieldPerEvent = []
        listYieldPerEventError = []
        listYieldPerSelectedEvent = []
        listYieldPerSelectedEventError = []
        listLumi = []
        listYieldPerLumi = []
        listYieldPerLumiError = []
        histNames = ["analysis-same-event-pairing/output;1/PairsBarrelSEPM_muonCandidateCut/MyMassPt", "analysis-event-selection/output;1/Event_BeforeCuts/VtxZ", "analysis-event-selection/output;1/Event_AfterCuts/VtxZ"]
        for irun, (run, rate) in enumerate(ir.items()):
            if run in excludedRuns:
                print(f"Skipping run {run}              ")
                continue
            print(f"Processing run {run} ({irun+1}/{len(ir)})...            ", end='\r')
            try:
                tmpHistDict = get_histograms_from_file(f"{dir}/{run}/AnalysisResults.root", histNames)
            except:
                continue
            tmpHistEvent = tmpHistDict["analysis-event-selection/output;1/Event_BeforeCuts/VtxZ"]
            tmpHistEventSelected = tmpHistDict["analysis-event-selection/output;1/Event_AfterCuts/VtxZ"]
            tmpHist = tmpHistDict["analysis-same-event-pairing/output;1/PairsBarrelSEPM_muonCandidateCut/MyMassPt"]
            tmpHistMassCoherent = tmpHist.ProjectionX(f"histMassCoherent_run{run}", firstybin=1, lastybin=4)
            # Calculate yield as pair counts in mass window -- impossible to get fit to every run
            lowBin = tmpHistMassCoherent.FindBin(2.95)
            highBin = tmpHistMassCoherent.FindBin(3.25) - 1
            nCandidates = tmpHistMassCoherent.Integral(lowBin, highBin)
            listYield.append(nCandidates)
            nCandidatesError = np.sqrt(nCandidates)
            listYieldPerEvent.append(nCandidates / tmpHistEvent.GetEntries())
            listYieldPerEventError.append((nCandidates / tmpHistEvent.GetEntries()) * (nCandidatesError / nCandidates))
            listYieldPerSelectedEvent.append(nCandidates / tmpHistEventSelected.GetEntries())
            listYieldPerSelectedEventError.append((nCandidates / tmpHistEventSelected.GetEntries()) * (nCandidatesError / nCandidates))
            listRun.append(run)
            listIr.append(rate)
            listEvents.append(tmpHistEvent.GetEntries())
            listEventsSelected.append(tmpHistEventSelected.GetEntries())
            lumi = self.histLumi.GetBinContent(self.histLumi.GetXaxis().FindBin(run))
            listLumi.append(lumi) 
            listYieldPerLumi.append(nCandidates / lumi)
            listYieldPerLumiError.append((nCandidates / lumi) * (nCandidatesError / nCandidates))
            self.runHists[run] = tmpHistMassCoherent
            del tmpHistDict
            del tmpHistEvent
            del tmpHist
            del tmpHistMassCoherent

        print(f"Processed the following {len(self.runHists)} runs: {self.runHists.keys()}")
        self.runDataFrame = pd.DataFrame({
            "run": listRun,
            "ir": listIr,
            "yield": listYield,
            "nEvents": listEvents,
            "nEventsSelected": listEventsSelected,
            "yieldPerEvent": listYieldPerEvent,
            "yieldPerEventError": listYieldPerEventError,
            "yieldPerSelectedEvent": listYieldPerSelectedEvent,
            "yieldPerSelectedEventError": listYieldPerSelectedEventError,
            "lumi": listLumi,
            "yieldPerLumi": listYieldPerLumi,
            "yieldPerLumiError": listYieldPerLumiError
        })
        self.runDataFrame = self.runDataFrame.sort_values("ir")

class Efficiency():
    def __init__(self, numeratorTitle, denominatorTitle, histNumerator, histDenominator):
        self.histNumerator = histNumerator
        self.histDenominator = histDenominator
        self.histogram = histNumerator.Clone()
        self.histogram.Divide(histNumerator, histDenominator, 1, 1, "B")
        self.histogram.SetName(f"eff_{histNumerator.GetName()}_{histDenominator.GetName()}")
        self.histogram.SetTitle(f"eff_{histNumerator.GetName()}_{histDenominator.GetName()}")
        self.title = "#frac{%s}{%s}" % (numeratorTitle, denominatorTitle)
        self.shortTitle = ""

    def draw_title(self, x, y, size=0.04, includeshort=False):
        self.latexTitle = r.TLatex()
        self.latexTitle.SetTextSize(size)
        self.latexTitle.SetTextAlign(22)
        if includeshort:
            self.latexTitle.DrawLatexNDC(x, y, self.title+"    "+self.shortTitle)
        else:
            self.latexTitle.DrawLatexNDC(x, y, self.title)

    def replace_displayed_title(self, includeshort=False, size=0.035):
        # Replace the displayed title of the histogram with the TeX formatted title
        self.histogram.SetTitle("")
        self.draw_title(0.5, 0.95, size=size, includeshort=includeshort)

    def take_ownership(self):
        self.histNumerator.SetDirectory(0)
        self.histDenominator.SetDirectory(0)
        self.histogram.SetDirectory(0)

class FactorizedEfficiency():
    def __init__(self, nFactors):
        if nFactors == 0:
            raise Exception("Need at least one factor!")
        self.nFactors = nFactors
        self.factors = []

    def add_factor(self, numeratorTitle, denominatorTitle, histNumerator, histDenominator):
        self.factors.append(Efficiency(numeratorTitle, denominatorTitle, histNumerator, histDenominator))
        self.factors[-1].shortTitle = f"({len(self.factors)})"

    def calculate_total_efficiency(self):
        if len(self.factors) != self.nFactors:
            raise Exception(f"Cannot calculate total efficiency: Only have {len(self.factors)} out of {self.nFactors} factors!")
        self.totalEfficiency = self.factors[0].histogram.Clone()
        self.totalEfficiency.SetName(f"totalEfficiency{self.nFactors}Factors")
        self.totalEfficiency.SetTitle(f"Total efficiency calculated from {self.nFactors} factors")
        self.totalEfficiencyTitle = self.factors[0].title
        self.totalEfficiencyShortTitle = self.factors[0].shortTitle
        for factor in self.factors[1:]:
            self.totalEfficiency.Multiply(self.totalEfficiency, factor.histogram, 1, 1, "B")
            self.totalEfficiencyTitle += "#times%s" % (factor.title)
            self.totalEfficiencyShortTitle += "#times%s" % (factor.shortTitle)

    def draw_title(self, x, y, size=0.04):
        self.totalEfficiencyLatexTitle = r.TLatex()
        self.totalEfficiencyLatexTitle.SetTextSize(size)
        self.totalEfficiencyLatexTitle.SetTextAlign(22)
        self.totalEfficiencyLatexTitle.DrawLatexNDC(x, y, self.totalEfficiencyTitle)

    def replace_displayed_title(self, size=None, short=False):
        # Replace the displayed title of the histogram with the TeX formatted title
        self.totalEfficiency.SetTitle("")
        if short:
            self.totalEfficiencyLatexTitle = r.TLatex()
            self.totalEfficiencyLatexTitle.SetTextSize(0.04)
            self.totalEfficiencyLatexTitle.SetTextAlign(22)
            self.totalEfficiencyLatexTitle.DrawLatexNDC(0.5, 0.95, self.totalEfficiencyShortTitle)
            return
        if size is None:
            self.draw_title(0.5, 0.95, 0.08/self.nFactors)
        else:
            self.draw_title(0.5, 0.95, size)

    def set_line_colors(self, color):
        self.totalEfficiency.SetLineColor(color)
        for factor in self.factors:
            factor.histogram.SetLineColor(color)

    def remove_stat_boxes(self):
        self.totalEfficiency.SetStats(0)
        for factor in self.factors:
            factor.histogram.SetStats(0)

    def draw(self, title="", labels=None, palette=default_palette, width=1400, height=800):
        r.gStyle.SetTitleFont(132, "")
        r.gStyle.SetLegendFont(132)
        colors = utils.create_seaborn_palette(palette)
        if labels is None:
            labels = [f.shortTitle for f in self.factors]
        
        if hasattr(self, 'canvas'):
            del self.canvas
        self.canvas = r.TCanvas('cFactorizedEff', 'cFactorizedEff', width, height)
        self.canvas.Divide(2,1,1e-12,0.01)

        self.canvas.cd(1)
        r.gPad.SetLeftMargin(0.1)
        r.gPad.SetRightMargin(0.05)
        self.legend = r.TLegend(0.5, 0.15, 0.89, 0.325)
        self.legend.SetBorderSize(0)
        self.legend.SetNColumns(2)
        for i, factor in enumerate(self.factors):
            factor.histogram.SetTitle(title)
            factor.histogram.GetYaxis().SetTitle("#varepsilon")
            factor.histogram.GetYaxis().SetRangeUser(0, 1)
            factor.histogram.Draw('SAME EP')
            factor.histogram.SetLineColor(colors[i].GetNumber())
            factor.histogram.SetMarkerStyle(marker_styles[i])
            factor.histogram.SetMarkerColor(colors[i].GetNumber())
            factor.histogram.SetStats(0)
            self.legend.AddEntry(factor.histogram, labels[i], 'p')
        self.legend.Draw()

        self.canvas.cd(2)
        self.legendCumulative = r.TLegend(0.12, 0.12, 0.899, 0.393)
        self.legendCumulative.SetBorderSize(0)
        self.legendCumulative.SetMargin(0.02)
        r.gPad.SetLeftMargin(0.10)
        r.gPad.SetRightMargin(0.05)
        name_string = "1"
        label = labels[0]
        self.cumulativeEfficiencies = [self.factors[0].histogram.Clone()]
        self.cumulativeEfficiencies[0].SetName(name_string)
        self.cumulativeEfficiencies[0].Draw('EP')
        self.cumulativeEfficiencies[0].SetLineColor(colors[0].GetNumber())
        self.cumulativeEfficiencies[0].SetMarkerStyle(marker_styles[0])
        self.cumulativeEfficiencies[0].SetMarkerColor(colors[0].GetNumber())
        self.cumulativeEfficiencies[0].SetStats(0)
        self.cumulativeEfficiencies[0].GetYaxis().SetRangeUser(0.004, 1)
        self.cumulativeEfficiencies[0].SetTitle(title)
        self.cumulativeEfficiencies[0].GetYaxis().SetTitle("#varepsilon")
        self.legendCumulative.AddEntry(self.cumulativeEfficiencies[0], labels[0], 'p')
        for i, factor in enumerate(self.factors[1:]):
            ce = self.cumulativeEfficiencies[i].Clone()
            ce.Multiply(factor.histogram)
            name_string += f"_{i+2}"
            self.cumulativeEfficiencies.append(ce)
            self.cumulativeEfficiencies[i+1].SetName(name_string)
            self.cumulativeEfficiencies[i+1].Draw('SAME EP')
            self.cumulativeEfficiencies[i+1].SetLineColor(colors[i+1].GetNumber())
            self.cumulativeEfficiencies[i+1].SetMarkerStyle(marker_styles[i+1])
            self.cumulativeEfficiencies[i+1].SetMarkerColor(colors[i+1].GetNumber())
            self.cumulativeEfficiencies[i+1].SetStats(0)
            label += " #times " + labels[i+1]
            self.legendCumulative.AddEntry(self.cumulativeEfficiencies[i+1], label, 'p')
            del ce
        self.legendCumulative.Draw()
        r.gPad.SetLogy()

        self.canvas.Draw()

class FitResult():
    "All information about the invariant mass fit to a pT bin"

    def __init__(self, lowerMass, upperMass, backgroundName, binIdx, isNominal=True):
        self.binIdx = binIdx
        self.isNominal = isNominal
        self.lowerMass = lowerMass
        self.upperMass = upperMass
        self.backgroundName = backgroundName

    def create_fit_results(self, reflectedLowerMass, reflectedUpperMass, bin, 
                           fitFunc, signalFunc, backgroundFunc, dataReflFunc, corrBkgFunc, totalBackgroundFunc,
                           doPrint=True):
        try:
            self.fitChi2Ndf = fitFunc.GetChisquare() / fitFunc.GetNDF()
        except:
            print("WARNING: Did not manage to calculate the chi2/ndf for the fit function!")
            self.fitChi2Ndf = 0
        self.fitMu = fitFunc.GetParameter('Mean')
        self.fitSigma = fitFunc.GetParameter('Sigma')
        self.nSignal = fitFunc.GetParameter(0) / bin.massPtSlice.GetBinWidth(1)
        self.nCombBackground = backgroundFunc.Integral(self.fitMu - 3*self.fitSigma, self.fitMu + 3*self.fitSigma) / bin.massPtSlice.GetBinWidth(1)
        if doPrint:
            print(f"bin {bin.index} nCombBackground = {self.nCombBackground} was calculated by integration from {self.fitMu - 3*self.fitSigma} to {self.fitMu + 3*self.fitSigma}")
        self.nReflBackground = dataReflFunc.GetParameter(0) / bin.massPtSlice.GetBinWidth(1)
        # Symmetric Hesse error
        self.relativeStatError = fitFunc.GetParError(0) / (bin.massPtSlice.GetBinWidth(1) * self.nSignal)
        # Asymmetric MINOS errors
        self.relativeStatErrorLower = np.abs(self.result.LowerError(0)) / (bin.massPtSlice.GetBinWidth(1) * self.nSignal)
        self.relativeStatErrorUpper = np.abs(self.result.UpperError(0)) / (bin.massPtSlice.GetBinWidth(1) * self.nSignal)
        signalIntegral3Sigma = signalFunc.Integral(self.fitMu - 3*self.fitSigma, self.fitMu + 3*self.fitSigma) / bin.massPtSlice.GetBinWidth(1)
        reflIntegral3Sigma = dataReflFunc.Integral(self.fitMu - 3*self.fitSigma, self.fitMu + 3*self.fitSigma) / bin.massPtSlice.GetBinWidth(1)
        corrIntegral3Sigma = (corrBkgFunc.Integral(self.fitMu - 3*self.fitSigma, self.fitMu + 3*self.fitSigma) / bin.massPtSlice.GetBinWidth(1)) if corrBkgFunc is not None else 0
        self.signalToBackground = signalIntegral3Sigma / (self.nCombBackground + reflIntegral3Sigma + corrIntegral3Sigma)
        self.signalSignificance = signalIntegral3Sigma / np.sqrt(signalIntegral3Sigma + self.nCombBackground + reflIntegral3Sigma + corrIntegral3Sigma)
        if doPrint:
            print(f"bin {bin.index} signal significance and S/B were calculated by integrating the signal and background components from {self.fitMu - 3*self.fitSigma} to {self.fitMu + 3*self.fitSigma}, yielding nSig={signalIntegral3Sigma}, nComb={self.nCombBackground}, nReflBackground (3#sigma)={reflIntegral3Sigma}")
        self.fitFunc = fitFunc
        self.signalFunc = signalFunc
        self.backgroundFunc = backgroundFunc
        self.corrBkgFunc = corrBkgFunc
        self.dataReflFunc = dataReflFunc
        self.totalBackgroundFunc = totalBackgroundFunc

class PtBin():
    """A single pT bin of an analysis"""

    def __init__(self, index, lowerPt, upperPt):
        self.index = index
        self.lowerPt = lowerPt
        self.upperPt = upperPt
        # Array to hold systematic fit variation results
        self.fitResults = []
        if self.lowerPt >= 6.:
            self.invMassRebin = 2
        else:
            self.invMassRebin = 1

    def draw(self, createCanvas=True):
        if createCanvas:
            if hasattr(self, 'canvas'):
                del self.canvas
            self.canvas = r.TCanvas(f"canvasPtBin{self.index}", f"canvasPtBin{self.index}")
        if hasattr(self, 'nominalFitResult'):
            self.legend = r.TLegend(0.15, 0.15, 0.4, 0.3)
            self.nominalFitResult.fitFunc.Draw()
            self.nominalFitResult.fitFunc.SetTitle(f"{self.lowerPt} < p_{{T}} < {self.upperPt} GeV/c")
            self.legend.AddEntry(self.nominalFitResult.fitFunc, "Total fit function")
            self.nominalFitResult.signalFunc.Draw("same LF2")
            self.legend.AddEntry(self.nominalFitResult.signalFunc, "Signal")
            self.nominalFitResult.backgroundFunc.Draw("same")
            self.legend.AddEntry(self.nominalFitResult.backgroundFunc, f"Comb. background ({self.nominalFitResult.backgroundName})")
            if self.nominalFitResult.corrBkgFunc is not None:
                self.nominalFitResult.corrBkgFunc.Draw("same")
                self.legend.AddEntry(self.nominalFitResult.corrBkgFunc, "D^{0}#rightarrow K^{-}#pi^{+}#pi^{0}")
            # self.nominalFitResult.totalBackgroundFunc.Draw("same")
            # Text box with info
            self.textBox = r.TPaveText(0.65, 0.5, 0.95, 0.85, "NDC")
            self.textBox.SetName(f"textbox_pTbin{self.index}")
            self.textBox.SetFillColor(0)
            self.textBox.SetFillStyle(0)
            self.textBox.SetBorderSize(0)
            self.textBox.SetTextAlign(12)
            self.textBox.SetTextFont(42)
            self.textBox.SetTextSize(0.04)
            self.textBox.AddText(f"#mu = {self.nominalFitResult.fitMu:.3f}")
            self.textBox.AddText(f"#sigma = {self.nominalFitResult.fitSigma:.4f}")
            self.textBox.AddText(f"S = {self.nominalFitResult.nSignal:.3f}")
            self.textBox.AddText(f"S/B (3#sigma)= {self.nominalFitResult.signalToBackground:.3f}")
            self.textBox.AddText(f"S/#sqrt{{S+B}} = {self.nominalFitResult.signalSignificance:.3f}")
            self.textBox.AddText(f"Refl/S = {self.nominalFitResult.nReflBackground/self.nominalFitResult.nSignal:.3f}")
            self.textBox.AddText(f"#chi^{{2}}/ndf = {self.nominalFitResult.fitChi2Ndf:.3f}")
            self.textBox.Draw()
            # Lines to show fitting range
            self.lineFitrangeLow = r.TLine(self.nominalFitResult.lowerMass, r.gPad.GetUymin(), self.nominalFitResult.lowerMass, r.gPad.GetUymax())
            self.lineFitrangeLow.SetLineColor(r.kBlack)
            self.lineFitrangeLow.SetLineStyle(2)
            self.lineFitrangeLow.SetLineWidth(1)
            self.lineFitrangeLow.Draw()
            self.lineFitrangeHigh = r.TLine(self.nominalFitResult.upperMass, r.gPad.GetUymin(), self.nominalFitResult.upperMass, r.gPad.GetUymax())
            self.lineFitrangeHigh.SetLineColor(r.kBlack)
            self.lineFitrangeHigh.SetLineStyle(2)
            self.lineFitrangeHigh.SetLineWidth(1)
            self.lineFitrangeHigh.Draw()
            self.massPtSlice.Draw("same E")
            self.massPtSlice.SetStats(0)
        else:
            self.massPtSlice.Draw("E")
            self.massPtSlice.SetStats(0)
        if createCanvas:
            self.canvas.Draw()
        
    def draw_reflected_fit(self, createCanvas=True):
        if createCanvas:
            if hasattr(self, 'canvasReflFit'):
                del self.canvasReflFit
            self.canvasReflFit = r.TCanvas("canvasReflFit", "canvasReflFit", 400, 333)
            self.canvasReflFit.cd()

        self.massPtSliceReflected.Draw("E")
        self.massPtSliceReflected.SetStats(0)
        self.reflFunc1 = r.TF1(f"fReflGauss1_bin{self.index}", fit_functions.signal, 1.3, 2.3, 3)
        self.reflFunc1.SetParameters(np.array([self.reflFunc.GetParameter(0) * self.reflFunc.GetParameter(1),
                                              self.reflFunc.GetParameter(2), self.reflFunc.GetParameter(3)], dtype='d'))
        self.reflFunc1.SetLineColor(r.kGreen)
        self.reflFunc1.SetLineStyle(r.kDashed)
        self.reflFunc1.Draw("same")
        self.reflFunc2 = r.TF1(f"fReflGauss2_bin{self.index}", fit_functions.signal, 1.3, 2.3, 3)
        self.reflFunc2.SetParameters(np.array([self.reflFunc.GetParameter(0) * (1 - self.reflFunc.GetParameter(1)),
                                              self.reflFunc.GetParameter(4), self.reflFunc.GetParameter(5)], dtype='d'))
        self.reflFunc2.SetLineColor(r.kMagenta)
        self.reflFunc2.SetLineStyle(r.kDashed)
        self.reflFunc2.Draw("same")
        self.reflFunc.SetLineColor(r.kRed)
        self.reflFunc.Draw("same")
        # Text box with info
        self.textBoxReflected = r.TPaveText(0.6, 0.4, 0.85, 0.9, "NDC")
        self.textBoxReflected.SetName(f"textboxreflected_bin{self.index}")
        self.textBoxReflected.SetFillColor(0)
        self.textBoxReflected.SetFillStyle(0)
        self.textBoxReflected.SetBorderSize(0)
        self.textBoxReflected.SetTextAlign(12)
        self.textBoxReflected.SetTextFont(42)
        self.textBoxReflected.SetTextSize(0.035)
        self.textBoxReflected.AddText(f"Refl/S = {self.reflectedRatio:.3f}")
        self.textBoxReflected.AddText(f"Integral = {self.reflFunc.GetParameter('Integral'):.3f}")
        self.textBoxReflected.AddText(f"RelNorm = {self.reflFunc.GetParameter('RelNorm'):.3f}")
        self.textBoxReflected.AddText(f"Mean1 = {self.reflFunc.GetParameter('Mean1'):.3f}")
        self.textBoxReflected.AddText(f"Sigma1 = {self.reflFunc.GetParameter('Sigma1'):.3f}")
        self.textBoxReflected.AddText(f"Mean2 = {self.reflFunc.GetParameter('Mean2'):.3f}")
        self.textBoxReflected.AddText(f"Sigma2 = {self.reflFunc.GetParameter('Sigma2'):.3f}")
        self.textBoxReflected.Draw()

        if createCanvas:
            self.canvasReflFits.Draw()

    def draw_systematic_fit_variations(self, showFits=False):
        if showFits:
            nCols = 5
            nPanels = len(self.fitResults) + 3
            nRows = int(np.ceil(nPanels/nCols))
        else:
            nCols = 3
            nRows = 1
        if hasattr(self, 'canvasSystematicFitVariations'):
            del self.canvasSystematicFitVariations
        self.canvasSystematicFitVariations = r.TCanvas("canvasSystematicFitVariations", "canvasSystematicFitVariations", nCols * 400, nRows * 333)
        self.canvasSystematicFitVariations.Divide(nCols, nRows, 0.002, 0.01)
        
        self.histSystematicYieldDistribution = r.TH1D("histSystematicYieldDistribution", f"{self.lowerPt} < p_{{T}} < {self.upperPt} GeV/c", 300, 0, 2*self.nominalFitResult.nSignal)
        self.histSystematicYieldDistribution.GetXaxis().SetTitle("Raw yield")
        self.histSystematicYieldDistribution.GetYaxis().SetTitle("Counts")
        self.histSystematicYieldVsTrial = r.TH1D("histSystematicYieldVsTrial", f"{self.lowerPt} < p_{{T}} < {self.upperPt} GeV/c", len(self.fitResults), 0, len(self.fitResults))
        self.histSystematicYieldVsTrial.GetXaxis().SetTitle("Trial")
        self.histSystematicYieldVsTrial.GetYaxis().SetTitle("Raw yield")
        self.histSystematicChi2ndfVsTrial = r.TH1D("histSystematicChi2ndfVsTrial", f"{self.lowerPt} < p_{{T}} < {self.upperPt} GeV/c", len(self.fitResults), 0, len(self.fitResults))
        self.histSystematicChi2ndfVsTrial.GetXaxis().SetTitle("Trial")
        self.histSystematicChi2ndfVsTrial.GetYaxis().SetTitle("#chi^{2}/ndf")
        self.textBoxesSystematicVariations = []
        self.linesFitrangeLow = []
        self.linesFitrangeHigh = []
        panelIndex = 1
        for i, fitResult in enumerate(self.fitResults):
            self.histSystematicYieldDistribution.Fill(fitResult.nSignal)
            self.histSystematicYieldVsTrial.SetBinContent(i+1, fitResult.nSignal)
            self.histSystematicYieldVsTrial.SetBinError(i+1, fitResult.nSignal*fitResult.relativeStatError)
            self.histSystematicChi2ndfVsTrial.SetBinContent(i+1, fitResult.fitChi2Ndf)
            if showFits:
                self.canvasSystematicFitVariations.cd(panelIndex)
                self.massPtSlice.Draw("E")
                self.massPtSlice.SetStats(0)
                fitResult.fitFunc.Draw("same")
                fitResult.backgroundFunc.Draw("same")
                fitResult.totalBackgroundFunc.Draw("same")
                # Text box with info
                self.textBoxesSystematicVariations.append(r.TPaveText(0.6, 0.5, 0.9, 0.85, "NDC"))
                self.textBoxesSystematicVariations[i].SetName(f"textbox_trial{i+1}")
                self.textBoxesSystematicVariations[i].SetFillColor(0)
                self.textBoxesSystematicVariations[i].SetFillStyle(0)
                self.textBoxesSystematicVariations[i].SetBorderSize(0)
                self.textBoxesSystematicVariations[i].SetTextAlign(12)
                self.textBoxesSystematicVariations[i].SetTextFont(42)
                self.textBoxesSystematicVariations[i].SetTextSize(0.04)
                self.textBoxesSystematicVariations[i].AddText(f"background: {fitResult.backgroundName}")
                self.textBoxesSystematicVariations[i].AddText(f"fit range: [{fitResult.lowerMass:.2f}, {fitResult.upperMass:.2f}]")
                self.textBoxesSystematicVariations[i].AddText(f"#mu = {fitResult.fitMu:.3f}")
                self.textBoxesSystematicVariations[i].AddText(f"#sigma = {fitResult.fitSigma:.3f}")
                self.textBoxesSystematicVariations[i].AddText(f"Refl/S = {fitResult.nReflBackground/fitResult.nSignal:.3f}")
                self.textBoxesSystematicVariations[i].AddText(f"S = {fitResult.nSignal:.3f}")
                self.textBoxesSystematicVariations[i].AddText(f"#chi^{{2}}/ndf = {fitResult.fitChi2Ndf:.3f}")
                self.textBoxesSystematicVariations[i].Draw()
                # Lines to show fitting range
                self.linesFitrangeLow.append(r.TLine(fitResult.lowerMass, r.gPad.GetUymin(), fitResult.lowerMass, r.gPad.GetUymax()))
                self.linesFitrangeLow[i].SetLineColor(r.kBlack)
                self.linesFitrangeLow[i].SetLineStyle(2)
                self.linesFitrangeLow[i].SetLineWidth(1)
                self.linesFitrangeLow[i].Draw()
                self.linesFitrangeHigh.append(r.TLine(fitResult.upperMass, r.gPad.GetUymin(), fitResult.upperMass, r.gPad.GetUymax()))
                self.linesFitrangeHigh[i].SetLineColor(r.kBlack)
                self.linesFitrangeHigh[i].SetLineStyle(2)
                self.linesFitrangeHigh[i].SetLineWidth(1)
                self.linesFitrangeHigh[i].Draw()
                panelIndex += 1
        self.canvasSystematicFitVariations.cd(panelIndex)
        self.histSystematicYieldVsTrial.Draw("E")
        self.lineNominalYield = r.TLine(self.histSystematicYieldVsTrial.GetXaxis().GetXmin(), self.nominalFitResult.nSignal, self.histSystematicYieldVsTrial.GetXaxis().GetXmax(), self.nominalFitResult.nSignal)
        self.lineNominalYield.SetLineColor(r.kRed)
        self.lineNominalYield.SetLineStyle(2)
        self.lineNominalYield.SetLineWidth(2)
        self.lineNominalYield.Draw('same')
        self.canvasSystematicFitVariations.cd(panelIndex+1)
        self.histSystematicChi2ndfVsTrial.SetMarkerStyle(20)
        self.histSystematicChi2ndfVsTrial.Draw("P")
        self.lineNominalChi2ndf = r.TLine(self.histSystematicChi2ndfVsTrial.GetXaxis().GetXmin(), self.nominalFitResult.fitChi2Ndf, self.histSystematicChi2ndfVsTrial.GetXaxis().GetXmax(), self.nominalFitResult.fitChi2Ndf)
        self.lineNominalChi2ndf.SetLineColor(r.kRed)
        self.lineNominalChi2ndf.SetLineStyle(2)
        self.lineNominalChi2ndf.SetLineWidth(2)
        self.lineNominalChi2ndf.Draw('same')
        self.canvasSystematicFitVariations.cd(panelIndex+2)
        self.histSystematicYieldDistribution.Draw("hist")
        self.textBoxRmsOverMean = r.TPaveText(0.6, 0.5, 0.9, 0.6, "NDC")
        self.textBoxRmsOverMean.AddText(f"RMS/#mu = {100*self.histSystematicYieldDistribution.GetRMS()/self.histSystematicYieldDistribution.GetMean():.2f}%")
        self.textBoxRmsOverMean.Draw()
        self.textBoxRmsOverMean.SetFillColor(0)
        self.textBoxRmsOverMean.SetFillStyle(0)
        self.textBoxRmsOverMean.SetBorderSize(0)
        self.textBoxRmsOverMean.SetTextAlign(12)
        self.textBoxRmsOverMean.SetTextFont(42)
        self.textBoxRmsOverMean.SetTextSize(0.04)
        self.canvasSystematicFitVariations.Draw()
        self.lineNominalYieldVert = r.TLine(self.nominalFitResult.nSignal, r.gPad.GetUymin(), self.nominalFitResult.nSignal, r.gPad.GetUymax())
        self.lineNominalYieldVert.SetLineColor(r.kRed)
        self.lineNominalYieldVert.SetLineStyle(2)
        self.lineNominalYieldVert.SetLineWidth(2)
        self.lineNominalYieldVert.Draw('same')

    def draw_fit_result(self, index):
        if hasattr(self, 'canvasFitResult'):
            del self.canvasFitResult
        self.canvasFitResult = r.TCanvas(f"canvasPtBin{self.index}FitResult", f"canvasPtBin{self.index}FitResult")
        self.fitResults[index].legend = r.TLegend(0.15, 0.15, 0.4, 0.3)
        self.fitResults[index].fitFunc.Draw()
        self.fitResults[index].legend.AddEntry(self.fitResults[index].fitFunc, "Total fit function")
        self.fitResults[index].signalFunc.Draw("same LF2")
        self.fitResults[index].legend.AddEntry(self.fitResults[index].signalFunc, "Signal")
        self.fitResults[index].backgroundFunc.Draw("same")
        self.fitResults[index].legend.AddEntry(self.fitResults[index].backgroundFunc, f"Comb. background ({self.fitResults[index].backgroundName})")
        if self.fitResults[index].corrBkgFunc is not None:
            self.fitResults[index].corrBkgFunc.Draw("same")
            self.fitResults[index].legend.AddEntry(self.fitResults[index].corrBkgFunc, "D^{0}#rightarrow K^{-}#pi^{+}#pi^{0}")
        self.fitResults[index].totalBackgroundFunc.Draw("same")
        # Text box with info
        self.fitResults[index].textBox = r.TPaveText(0.65, 0.5, 0.95, 0.85, "NDC")
        self.fitResults[index].textBox.SetName(f"textbox_pTbin{self.index}FitResult")
        self.fitResults[index].textBox.SetFillColor(0)
        self.fitResults[index].textBox.SetFillStyle(0)
        self.fitResults[index].textBox.SetBorderSize(0)
        self.fitResults[index].textBox.SetTextAlign(12)
        self.fitResults[index].textBox.SetTextFont(42)
        self.fitResults[index].textBox.SetTextSize(0.04)
        self.fitResults[index].textBox.AddText(f"#mu = {self.fitResults[index].fitMu:.3f}")
        self.fitResults[index].textBox.AddText(f"#sigma = {self.fitResults[index].fitSigma:.4f}")
        self.fitResults[index].textBox.AddText(f"S = {self.fitResults[index].nSignal:.3f}")
        self.fitResults[index].textBox.AddText(f"S/B (3#sigma)= {self.fitResults[index].signalToBackground:.3f}")
        self.fitResults[index].textBox.AddText(f"S/#sqrt{{S+B}} = {self.fitResults[index].signalSignificance:.3f}")
        self.fitResults[index].textBox.AddText(f"Refl/S = {self.fitResults[index].nReflBackground/self.fitResults[index].nSignal:.3f}")
        self.fitResults[index].textBox.AddText(f"#chi^{{2}}/ndf = {self.fitResults[index].fitChi2Ndf:.3f}")
        self.fitResults[index].textBox.Draw()
        # Lines to show fitting range
        self.fitResults[index].lineFitrangeLow = r.TLine(self.fitResults[index].lowerMass, r.gPad.GetUymin(), self.fitResults[index].lowerMass, r.gPad.GetUymax())
        self.fitResults[index].lineFitrangeLow.SetLineColor(r.kBlack)
        self.fitResults[index].lineFitrangeLow.SetLineStyle(2)
        self.fitResults[index].lineFitrangeLow.SetLineWidth(1)
        self.fitResults[index].lineFitrangeLow.Draw()
        self.fitResults[index].lineFitrangeHigh = r.TLine(self.fitResults[index].upperMass, r.gPad.GetUymin(), self.fitResults[index].upperMass, r.gPad.GetUymax())
        self.fitResults[index].lineFitrangeHigh.SetLineColor(r.kBlack)
        self.fitResults[index].lineFitrangeHigh.SetLineStyle(2)
        self.fitResults[index].lineFitrangeHigh.SetLineWidth(1)
        self.fitResults[index].lineFitrangeHigh.Draw()
        self.massPtSlice.Draw("same E")
        self.massPtSlice.SetStats(0)
        self.canvasFitResult.Draw()
        

class Analysis():
    """Class containing everything needed to calculate a D0 cross section from O2Physics output"""

    def __init__(self, pathTableMaker, dirData, dirMc, dirGen = None, dirRefl = None,
                 ptBins = [0., 0.75, 1.5, 2.25, 3., 4., 6., 8., 12.], rapidityRange = [-0.9, 0.9],
                 runList = None, hLumiPath=['eventselection-run3', 'luminosity', 'hLumiTCEafterBCcuts'], 
                 dataFileStructure=FileStructures.FLAT,
                 recFileStructure=FileStructures.FLAT,
                 genFileStructure=FileStructures.FLAT,
                 rapidityGapSelectionEfficiency = None,
                 pathHfTree = '/media/sigurd/T7/mc/HF_LHC25e4_All/train586890/mergedAO2D.root',
                 **kwargs):
        # The runList defines which files are looped over in run-by-run calculations
        self.runList = runList if runList is not None else DEFAULT_RUNLIST_PASS4.copy()
        self.runList = sorted(self.runList)
        print(f"This analysis contains {len(self.runList)} runs")

        # Import table-maker merged output file and check that it contains the correct runs
        self.fileTableMaker = r.TFile.Open(pathTableMaker)
        self.hLumiPath = hLumiPath
        self.check_file_for_runs(self.fileTableMaker, self.hLumiPath)

        self.dirData = dirData
        self.dirRec = dirMc
        self.dirGen = dirGen
        self.dirRefl = dirRefl
        self.pathHfTree = pathHfTree

        # Get default cut names, replace if specified in construction
        cutNames = {
            "kaonLegCutName" : "kaonPIDTPCTOFpTDCAz",
            "pionLegCutName" : "pionNoPIDpTDCAz",
            "pairCutName"    : "PtDepTauxyzprojCut"
        }
        cutNames.update(kwargs)
        self.kaonLegCutName = cutNames["kaonLegCutName"]
        self.pionLegCutName = cutNames["pionLegCutName"]
        self.pairCutName = cutNames["pairCutName"]
        if self.pairCutName != "":
            self.pairCutName = "_" + self.pairCutName
        # pT bins to be used in the differential cross section
        self.ptBinsArray = ptBins
        self.ptBins = []
        for i in range(len(self.ptBinsArray) - 1):
            self.ptBins.append(PtBin(i, self.ptBinsArray[i], self.ptBinsArray[i+1]))
        # Range of rapidity bin to be used (fiducial acceptance cut)
        self.minY = rapidityRange[0]
        self.maxY = rapidityRange[1]
        # Get the mass histograms needed for the signal extraction
        self.dataFileStructure = dataFileStructure
        self.recFileStructure = recFileStructure
        self.genFileStructure = genFileStructure
        self.prepare_histograms()
        # Get the integrated luminosity from the luminosity task
        self.calculate_lumi() # 1/µb
        print(f"Integrated luminosity (TCE trigger, after BC cuts): {self.lumi} 1/µb")
        self.runByRunLumi = {}
        for run in self.runList:
           self.runByRunLumi[run] = self.histLumi.GetBinContent(self.histLumi.GetXaxis().FindBin(run)) 
        self.runByRunLumiTotal = sum(self.runByRunLumi.values())
        if self.runByRunLumiTotal < self.lumi:
            print(f"WARNING: Lumi calculated as sum over runlist is not equal to the integral of lumi histogram! Setting lumi for this analysis to {self.runByRunLumiTotal} 1/µb")
            self.lumi = self.runByRunLumiTotal
        # Set the range of the mass axis for plotting and integrating reflected background
        self.massRange = [0., 4.]
        self.reweighting = False
        # Manually set the efficiency due to rapidity gap selection
        self.rapidityGapSelectionEfficiency = rapidityGapSelectionEfficiency
        if self.rapidityGapSelectionEfficiency is not None:
            print("INFO: Efficiency due to rapidity gap event selection was provided manually. Make sure the MC files do NOT contain a rapidity gap selection!")

    def calculate_lumi(self):
        self.lumi = 0
        ax = self.histLumi.GetXaxis()
        for i in range(1, self.histLumi.GetNbinsX() + 1):
            if ax.GetBinLabel(i) in self.runList:
                self.lumi += self.histLumi.GetBinContent(i)

    def correct_lumi(self, dir):
        for irun, run in enumerate(self.runList):
            print(f"Checking run {run} ({irun+1}/{len(self.runList)})...            ", end='\r')
            pathTableMaker = f"{dir}/{run}/mergedAnalysisResults.root"
            histTableMaker = get_histograms_from_file(pathTableMaker, ["table-maker/output;1/Event_AfterCuts/VtxZ"])["table-maker/output;1/Event_AfterCuts/VtxZ"]
            nEventsTableMaker = histTableMaker.GetEntries()
            pathTableReader = f"{self.dirData}/{run}/AnalysisResults.root"
            histTableReader = get_histograms_from_file(pathTableReader, ["analysis-event-selection/output;1/Event_BeforeCuts/VtxZ"])["analysis-event-selection/output;1/Event_BeforeCuts/VtxZ"]
            nEventsTableReader = histTableReader.GetEntries()
            correctionFactor = nEventsTableReader / nEventsTableMaker
            del histTableReader
            del histTableMaker
            if correctionFactor != 1:
                lumiThisRun = self.histLumi.GetBinContent(self.histLumi.GetXaxis().FindBin(run))
                lumiReduction = lumiThisRun * (1 - correctionFactor)
                print(f"\nRun {run} ({lumiThisRun:.2f} 1/µb): Only {100 * correctionFactor:.3f}% of events included. Reducing the analysis luminosity by {lumiReduction:.3f} 1/µb")
                self.lumi -= lumiReduction
        print(f"Done! New lumi = {self.lumi:.2f} 1/µb           ")

    def check_file_for_runs(self, file, strings):
        if len(strings) == 3:
            group, subGroup, histName = strings
            hist = file.Get(group).Get(subGroup).Get(histName)
        elif len(strings) == 2:
            group, histName = strings
            hist = file.Get(group).Get(histName)
        labels = []
        for bin in range(1, hist.GetNbinsX() + 1):
            labels.append(hist.GetXaxis().GetBinLabel(bin))
        # Remove empty labels
        labels = [label for label in labels if label]
        if sorted(labels) != sorted(self.runList):
            uniqueRuns = set(self.runList) - set(labels)
            uniqueLabels = set(labels) - set(self.runList)
            print(f"Items unique to the runList: {uniqueRuns}")
            print(f"Items unique to the file: {uniqueLabels}")
            if set(self.runList) <= set(labels):
                print(f"WARNING: File '{file.GetName()}' contains a superset of the runs in the runlist! Remember to correct the luminosity!")
            else:
                raise Exception(f"File '{file.GetName()}' does not contain the correct set of runs!")

    def prepare_histograms(self):
        self.groupNameD0Generated = "MCTruthGenAfterBcCuts_D0FS"
        self.groupNameD0PtMatched = f"PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4{self.pairCutName}_KPiFromD0FS"

        self.histLumi = self.fileTableMaker.Get(self.hLumiPath[0]).Get(self.hLumiPath[1])
        if len(self.hLumiPath) == 3:
            self.histLumi = self.histLumi.Get(self.hLumiPath[2])

        # --- Define lists of histogram names ---
        # Data histograms
        fullNameHistD0MassPt = "analysis-asymmetric-pairing/output;1/" + f"PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}{self.pairCutName}" + "/MyMassPtHisto"
        fullNameHistD0MassPtY = "analysis-asymmetric-pairing/output;1/" + f"PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}{self.pairCutName}" + "/MyMassPtYHisto"
        fullNameHistD0MassPtIsGap = "analysis-asymmetric-pairing/output;1/" + f"PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}{self.pairCutName}" + "/MyMassPtIsGapHisto"
        fullNameHistMultiDimA = "analysis-asymmetric-pairing/output;1/" + f"PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}{self.pairCutName}" + "/MyMassPtVtxNContribRealGapAHisto"
        fullNameHistMultiDimC = "analysis-asymmetric-pairing/output;1/" + f"PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}{self.pairCutName}" + "/MyMassPtVtxNContribRealGapCHisto"
        fullNameHistVtxNContribAfterCuts = "analysis-event-selection/output;1/Event_AfterCuts/VtxNContrib"
        fullNameHistVtxNContribRealAfterCuts = "analysis-event-selection/output;1/Event_AfterCuts/VtxNContribReal"
        # Main gen. lvl. histogram used for efficiency, and also all the histograms used for factorized efficiencies later
        fullNamesGen = ["MCTruthGenRec_D0FS", "MCTruthGenSel_D0FS", "MCTruthGenSelDaughtersInAcc_KPiFromD0FS", "MCTruthGenRecDaughtersInAcc_KPiFromD0FS", "MCTruthGenAfterBcCutsDaughtersInAcc_KPiFromD0FS"]
        fullNamesGen = ["analysis-asymmetric-pairing/output;1/" + name + "/PtMC_YMC" for name in fullNamesGen]
        fullNameHistD0PtYGenerated = "analysis-asymmetric-pairing/output;1/" + self.groupNameD0Generated + "/MyMcPtYHisto"
        fullNamesGen.append(fullNameHistD0PtYGenerated)
        # Main rec. matched histogram, the reflected histogram, and the rec. lvl. histograms used for factorized efficiencies later
        baseNamesMc = ["noTrackCut:noTrackCut", f"{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4", f"{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4{self.pairCutName}"]
        fullNamesMc = ["analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_" + baseName + "_KPiFromD0FS/Y_PtFine" for baseName in baseNamesMc]
        fullNamesMc += ["analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_" + baseName + "_KPiFromD0FS/Pt" for baseName in baseNamesMc]
        groupNameD0Matched = "analysis-asymmetric-pairing/output;1/" + self.groupNameD0PtMatched
        fullNameHistD0YPtMatched = "analysis-asymmetric-pairing/output;1/" + self.groupNameD0PtMatched + "/Y_PtFine"
        fullNameHistD0PtMatched = "analysis-asymmetric-pairing/output;1/" + self.groupNameD0PtMatched + "/Pt"
        fullNamesMc.append(fullNameHistD0YPtMatched)
        fullNameHistD0MassPtReflected = "analysis-asymmetric-pairing/output;1/" + f"{self.groupNameD0PtMatched}Reflected" + "/MyMassPtHisto"
        # Histograms needed for rescaling the IR dependence of the efficiency
        fullNamesGen.append("analysis-asymmetric-pairing/output;1/MCTruthGenAfterBcCuts_D0FS/MyMcPtYInteractionRateHisto")
        fullNamesMc.append(f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4{self.pairCutName}_KPiFromD0FS/MyYPtInteractionRateHisto")
        # Histogram used in factorized efficiency
        fullNamesMc.append("analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_beforeEvtCuts_noTrackCut:noTrackCut_KPiFromD0FS/Pt")
        fullNamesMc.append("analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_beforeEvtCuts_noTrackCut:noTrackCut_KPiFromD0FS/Y_PtFine")
        fullNamesMc.append("analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_D0KaonTPC:noTrackCut_KPiFromD0FS/Pt")
        fullNamesMc.append("analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_D0KaonTPC:noTrackCut_KPiFromD0FS/Y_PtFine")
        fullNamesMc.append("analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_D0KaonTPCTOF:noTrackCut_KPiFromD0FS/Pt")
        fullNamesMc.append("analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_D0KaonTPCTOF:noTrackCut_KPiFromD0FS/Y_PtFine")

        # Get the histograms from the AnalysisResults.root files
        self.dictMcHists = {}
        if self.dirGen is not None:
            self.dictGenHists = get_histograms(self.dirGen, self.runList, fullNamesGen, histogramNamesfileStructure=self.genFileStructure)
            print("INFO: Separate directory for MC files with generator lvl. histograms was specified")
        else:
            fullNamesMc += fullNamesGen
            self.dictGenHists = self.dictMcHists

        if self.dirRefl is not None:
            # Get the reflected histogram (and the reference not reflected histogram) from a separate file
            print("INFO: Separate directory for MC files with reflected histogram was specified")
            dictReflHist = get_histograms(self.dirRefl, self.runList, [fullNameHistD0MassPtReflected, fullNameHistD0PtMatched], histogramNamesfileStructure=self.recFileStructure)
        else:
            fullNamesMc.append(fullNameHistD0MassPtReflected)
            fullNamesMc.append(fullNameHistD0PtMatched)
            dictReflHist = self.dictMcHists

        # This dict always contains reco lvl. histograms, sometimes the reflected histogram, sometimes the gen lvl. histograms
        self.dictMcHists.update(get_histograms(self.dirRec, self.runList, fullNamesMc, histogramNamesfileStructure=self.recFileStructure))
        self.dictRecHists = self.dictMcHists

        # Fetch the histograms from the dicts
        # TODO: In principle these should also be projected inside the fiducial rapidity cut, but it will be a small effect
        self.histD0MassPtReflected = dictReflHist[fullNameHistD0MassPtReflected]
        self.histD0PtNotReflected = dictReflHist[fullNameHistD0PtMatched]

        try:
            self.histD0YPtMatched = self.dictRecHists[fullNameHistD0YPtMatched]
        except:
            self.histD0YPtMatched = self.dictRecHists[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4{self.pairCutName}_KPiFromD0FS/MyYPtInteractionRateHisto"]

        self.histD0PtMatched = project_fiducial_acceptance(self.histD0YPtMatched, self.minY, self.maxY)
        self.histD0PtMatched.SetName("histD0PtMatched")
        self.histD0PtMatched.SetTitle(f"Reconstructed, matched D0 in {self.minY:.1f} < y < {self.maxY:.1f}")

        self.histD0PtYGenerated = self.dictGenHists[fullNameHistD0PtYGenerated]
        # Project out our dy bin from the gen histogram
        lowerYBin = self.histD0PtYGenerated.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenerated.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenerated = self.histD0PtYGenerated.ProjectionX(f"projPtMcGen_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenerated.SetTitle(f"Generated D0 after BC cuts in {self.minY:.1f} < y < {self.maxY:.1f}")

        # Get data histograms
        tmpDict = get_histograms(self.dirData, self.runList, [fullNameHistD0MassPt, fullNameHistD0MassPtY, fullNameHistD0MassPtIsGap, fullNameHistVtxNContribAfterCuts, fullNameHistVtxNContribRealAfterCuts, fullNameHistMultiDimA, fullNameHistMultiDimC], histogramNamesfileStructure=self.dataFileStructure)
        try:
            self.histD0MassPtY = tmpDict[fullNameHistD0MassPtY]
            self.histD0MassPtY.GetZaxis().SetRangeUser(self.minY, self.maxY)
            self.histD0MassPt = self.histD0MassPtY.Project3D('yx') # x: mass, y: pT
        except:
            if (self.minY == -0.9 and self.maxY == 0.9):
                print("INFO: Did not find data mass vs pT vs y histogram, but rapidity range is [-0.9, 0.9]. Falling back to mass vs pT or mass vs pT vs IsGap histogram")
                try:
                    self.histD0MassPt = tmpDict[fullNameHistD0MassPt]
                except:
                    hist3d = tmpDict[fullNameHistD0MassPtIsGap]
                    self.histD0MassPt = hist3d.Project3D('yx')
                    del hist3d
            else:
                raise Exception(f"Unable to find suitable histogram in data to make projection in rapidity!")

        self.histVtxNContribAfterCuts = tmpDict[fullNameHistVtxNContribAfterCuts]
        self.histVtxNContribRealAfterCuts = tmpDict[fullNameHistVtxNContribRealAfterCuts]
        self.nEvents = self.histVtxNContribAfterCuts.GetEntries()
        del tmpDict

        # Get info and histograms belonging to the pT bins in the analysis 
        for i, bin in enumerate(self.ptBins):
            # Create an array of slices of the mass vs pT histogram and the MC reflected mass vs pT histogram
            lowerBin = self.histD0MassPt.GetYaxis().FindBin(bin.lowerPt)
            upperBin = self.histD0MassPt.GetYaxis().FindBin(bin.upperPt) - 1
            hProjectionMass = self.histD0MassPt.ProjectionX(f"projMass_{bin.lowerPt}_{bin.upperPt}", firstybin=lowerBin, lastybin=upperBin)
            hProjectionMass.Rebin(bin.invMassRebin)
            bin.massPtSlice = hProjectionMass

            hProjectionMassReflected = self.histD0MassPtReflected.ProjectionX(f"projMassMcReflected_{bin.lowerPt}_{bin.upperPt}", firstybin=lowerBin, lastybin=upperBin)
            hProjectionMassReflected.Rebin(7)
            hProjectionMassReflected.SetTitle(f"MC matched reflected Kpi invariant mass, {bin.lowerPt} <= pT < {bin.upperPt} GeV/c")
            bin.massPtSliceReflected = hProjectionMassReflected

            # Count the number of matched D0 in this bin's pT range
            matchedLowerBin = self.histD0PtMatched.GetXaxis().FindBin(bin.lowerPt)
            matchedUpperBin = self.histD0PtMatched.GetXaxis().FindBin(bin.upperPt) - 1
            bin.nMatched = self.histD0PtMatched.Integral(matchedLowerBin, matchedUpperBin)
            bin.nNotReflected = self.histD0PtNotReflected.Integral(matchedLowerBin, matchedUpperBin)

    def systematic_fit_variation(self, bin, backgroundDict, lowers, uppers, signalMuRange, signalSigmaRange, useCorrBkgTemplate=False, maxChi2=2.0, doPrint=False):
        if len(self.ptBins[bin].fitResults) != 0:
            print(f"WARNING: This bin already has {len(self.ptBins[bin].fitResults)}, these will now be overwritten!")
            self.ptBins[bin].fitResults = []
        reflRatioFactors = [1.0, 0.9, 1.1]
        signalWidthFactors = [1.0, 0.95, 1.05]
        nTrials = len(backgroundDict) * len(lowers) * len(uppers) * len(reflRatioFactors) * len(signalWidthFactors)
        counter = 0
        for backgroundFunction, initialParams in backgroundDict.items():
            for lower in lowers:
                for upper in uppers:
                    for reflRatioFactor in reflRatioFactors:
                        for signalWidthFactor in signalWidthFactors:
                            counter += 1
                            infoString = ""
                            infoString += f"Trial {counter}/{nTrials}: "
                            if signalWidthFactor != 1.0:
                                sigma = sigmaFree * signalWidthFactor 
                                infoString += f"Fixing sigma = {sigma:.3f}, "
                            else:
                                sigma = None
                                infoString += "Free sigma, "
                            if (lower == self.ptBins[bin].nominalFitResult.lowerMass and
                                upper == self.ptBins[bin].nominalFitResult.upperMass and
                                backgroundFunction.__name__ == self.ptBins[bin].nominalFitResult.backgroundName and
                                reflRatioFactor == 1.0 and signalWidthFactor == 1.0):
                                # This is the nominal result, skip it
                                print("This is the nominal settings, skipping!                                                                                                              ", end='\r')
                                continue
                            infoString += f"{backgroundFunction.__name__}, fit range [{lower}, {upper}], reflRatio factor = {reflRatioFactor}, "
                            fitResult = self.fit_inv_mass(bin, backgroundFunction, fitRange=[lower, upper], doPrint=doPrint, isNominal=False, 
                                                          signalMuRange=signalMuRange, signalSigma=sigma, signalSigmaRange=signalSigmaRange,
                                                          backgroundInitialParams=initialParams, reflRatioFactor=reflRatioFactor, useCorrBkgTemplate=useCorrBkgTemplate)
                            infoString += f"S = {fitResult.nSignal:.3f}, chi2/ndf = {fitResult.fitChi2Ndf:.3f}, "
                            # If this was the 'free' sigma case, save the sigma as reference for the +-5% variations
                            if signalWidthFactor == 1.0:
                                sigmaFree = fitResult.fitSigma
                            if fitResult.fitChi2Ndf > maxChi2:
                                infoString += f"chi2/ndf > {maxChi2}, trial rejected!              "
                            else:
                                self.ptBins[bin].fitResults.append(fitResult)
                                infoString += f"Trial saved!                                       "
                            print(infoString, end='\r')
        print(f"{nTrials} trials done!                                                                                                                                                                        ")

    def fit_reflected(self, bin, fitResult, doPrint=True):
        # Fit the reflected MC histogram with a double Gaussian
        reflFunc = r.TF1(f"fDoubleGauss_bin{bin}", fit_functions.reflected_background, self.massRange[0], self.massRange[1], 6)
        reflFunc.SetNpx(1000)
        reflFunc.SetParameters(1, 0.5, self.ptBins[bin].massPtSliceReflected.GetMean(), self.ptBins[bin].massPtSliceReflected.GetRMS()/2, self.ptBins[bin].massPtSliceReflected.GetMean(), self.ptBins[bin].massPtSliceReflected.GetRMS()/2)
        reflFunc.SetParNames("Integral", "RelNorm", "Mean1", "Sigma1", "Mean2", "Sigma2")
        # Require the integral to be positive
        reflFunc.SetParLimits(0, 0, 1e6)
        # Require the relative normalization to be a number between 0 and 1
        reflFunc.SetParLimits(1, 0, 1)
        # Require reasonable peak positions and widths for the Gaussians
        reflFunc.SetParLimits(2, self.ptBins[bin].massPtSliceReflected.GetXaxis().GetXmin(), self.ptBins[bin].massPtSliceReflected.GetXaxis().GetXmax())
        reflFunc.SetParLimits(4, self.ptBins[bin].massPtSliceReflected.GetXaxis().GetXmin(), self.ptBins[bin].massPtSliceReflected.GetXaxis().GetXmax())
        reflFunc.SetParLimits(3, 0.001, 0.1)
        reflFunc.SetParLimits(5, 0.001, 0.1)
        if doPrint:
            print("---- Fitting MC matched, reflected ----")
        fitResult.resultReflected = self.ptBins[bin].massPtSliceReflected.Fit(reflFunc, "LESR0" + ("Q" if not doPrint else ""))
        self.ptBins[bin].reflFunc = reflFunc

        self.ptBins[bin].reflectedRatio = self.ptBins[bin].massPtSliceReflected.GetEntries() / self.ptBins[bin].nNotReflected

    def fit_inv_mass(self, bin, backgroundFunction, fitRange = [1.64, 2.08], doPrint=True, isNominal=True, 
                     signalMuRange=None, signalSigmaRange=None, signalSigma=None,
                     backgroundInitialParams=None,
                     simpleFitInitialParams=[100, 1.85, 0.015, 100, 0],
                     simpleFitRange=[1.75, 1.98],
                     shadowRange=[1.8, 1.9],
                     useCorrBkgTemplate=False,
                     reflRatioFactor=1.0):
        # Create the results object to save info on this fit
        fitResult = FitResult(fitRange[0], fitRange[1], backgroundFunction.__name__, bin)
        if doPrint:
            print(f"====== Fitting bin {bin} ({self.ptBins[bin].lowerPt} < pT < {self.ptBins[bin].upperPt} GeV/c) ======")

        # Do the fit to the MC reflected background if needed
        if not hasattr(self.ptBins[bin], 'reflectedRatio'):
            self.fit_reflected(bin, fitResult, doPrint)
        reflFunc = self.ptBins[bin].reflFunc

        backgroundNPar = fit_functions.background_npar[backgroundFunction.__name__]
        if backgroundInitialParams is not None and len(backgroundInitialParams) != backgroundNPar:
            print(f"WARNING: {len(backgroundInitialParams)} initial parameters for combinatorial background specified, but '{backgroundFunction.__name__}' takes {backgroundNPar} parameters!")
        # Get the complete function to be used for the fit
        if useCorrBkgTemplate:
            # If requested, get a RDataFrame containing MC candidatates from a HF task output, to be used for correlated template background
            if self.pathHfTree is not None:
                if not hasattr(self.ptBins[bin], "histCorrBkg"):
                    df = utils.read_hf_tree(self.pathHfTree)
                    self.ptBins[bin].histCorrBkg = utils.get_mass_histogram_Kpipi0(df, self.ptBins[bin].lowerPt, self.ptBins[bin].upperPt)
                    self.ptBins[bin].histCorrBkg.Scale(1.0 / self.ptBins[bin].histCorrBkg.Integral())
                fitFunc = fit_functions.make_fit_model_with_corr_bkg(backgroundFunction, backgroundNPar, fitRange[0], fitRange[1], self.ptBins[bin].histCorrBkg)
                # Require the correlated background component to be non-negative
                nCorrIdx = fitFunc.GetParNumber("nCorr")
                fitFunc.SetParLimits(nCorrIdx, 0, 1e10)
            else:
                raise Exception("Requested templated background for D0->Kpipi0 but no HF root tree provided!")
        else:
            fitFunc = fit_functions.make_fit_model(backgroundFunction, backgroundNPar, fitRange[0], fitRange[1])
        fitFunc.SetName(f"fitFunc_bin{bin}")
        # Fix the parameters of the reflected components
        # TODO: These parameters should have some errors associated with them, coming from to the fit to MC
        fitFunc.FixParameter(3, self.ptBins[bin].reflectedRatio * reflRatioFactor) # Ratio refl/S
        if doPrint:
            print(f"Using reflectedRatio={self.ptBins[bin].reflectedRatio * reflRatioFactor} in fit")
        fitFunc.FixParameter(4, reflFunc.GetParameter("RelNorm")) # Relative normalization of the two Gaussians
        fitFunc.FixParameter(5, reflFunc.GetParameter("Mean1"))
        fitFunc.FixParameter(6, reflFunc.GetParameter("Sigma1"))
        fitFunc.FixParameter(7, reflFunc.GetParameter("Mean2"))
        fitFunc.FixParameter(8, reflFunc.GetParameter("Sigma2"))
        # Use a simple linear background to get initial parameters for the signal function
        simpleFitFunc = r.TF1(f"simpleFitFunc_bin{bin}", fit_functions.simple_fit_model, 1.7, 2.03, npar=5, ndim=1)
        simpleFitFunc.SetParameters(*simpleFitInitialParams)
        # Require yield to be positive
        simpleFitFunc.SetParLimits(0, 0, 100000)
        # If specified, set range for the signal mean
        if signalMuRange is not None:
            simpleFitFunc.SetParLimits(1, signalMuRange[0], signalMuRange[1])
        # If specified, set range for the signal width
        if signalSigmaRange is not None:
            simpleFitFunc.SetParLimits(2, signalSigmaRange[0], signalSigmaRange[1])
        if signalSigma is not None:
            simpleFitFunc.FixParameter(2, signalSigma)
        self.ptBins[bin].massPtSlice.Fit(simpleFitFunc, "LME0" + ("Q" if not doPrint else ""), "", simpleFitRange[0], simpleFitRange[1])
        fitFunc.SetParameter(0, simpleFitFunc.GetParameter(0)) # Signal yield
        fitFunc.SetParameter(1, simpleFitFunc.GetParameter(1)) # Mean
        fitFunc.SetParameter(2, simpleFitFunc.GetParameter(2)) # Sigma
        # Set the background parameters to safe defaults (avoid division by 0)
        for i in range(backgroundNPar):
            fitFunc.SetParameter(3 + 6 + i, 1.0 if backgroundInitialParams is None else backgroundInitialParams[i])
        if doPrint:
            print("Initially, the combinatorial background function parameters are:")
            for i in range(backgroundNPar):
                print(f"par[{3 + 6 + i}] = {fitFunc.GetParameter(3 + 6 + i)}")
        # Obtain a shadowed histogram with signal region removed, and do an initial fit to the background
        self.ptBins[bin].massPtSliceShadowed = self.ptBins[bin].massPtSlice.Clone(f"invMassShadowed_bin{bin}")
        for i in range(1, self.ptBins[bin].massPtSliceShadowed.GetNbinsX() + 1):
            binCenter = self.ptBins[bin].massPtSliceShadowed.GetBinCenter(i)
            if binCenter > shadowRange[0] and binCenter < shadowRange[1]:
                self.ptBins[bin].massPtSliceShadowed.SetBinContent(i, 0.0)
                self.ptBins[bin].massPtSliceShadowed.SetBinError(i, 0.0)
        # Fix the signal yield, mean and sigma to the previously obtained values
        fitFunc.FixParameter(0, simpleFitFunc.GetParameter(0)) # Signal yield
        fitFunc.FixParameter(1, simpleFitFunc.GetParameter(1)) # Mean
        fitFunc.FixParameter(2, simpleFitFunc.GetParameter(2)) # Sigma
        self.ptBins[bin].massPtSliceShadowed.Fit(fitFunc, "LME0" + ("Q" if not doPrint else ""), "", fitRange[0], fitRange[1])
        if doPrint:
            print("After fit to shadowed histogram, the combinatorial background function parameters are:")
            for i in range(backgroundNPar):
                print(f"par[{3 + 6 + i}] = {fitFunc.GetParameter(3 + 6 + i)}")
        # The parameters for the combinatorial background should now have reasonable values. Now release the signal function parameters for the final fit
        fitFunc.ReleaseParameter(0) # Signal yield
        fitFunc.ReleaseParameter(1) # Mean
        fitFunc.ReleaseParameter(2) # Sigma
        # Require yield to be positive
        fitFunc.SetParLimits(0, 0, 100000)
        # If specified, set range for the signal mean
        if signalMuRange is not None:
            fitFunc.SetParLimits(1, signalMuRange[0], signalMuRange[1])
        # If specified, set range for the signal width
        if signalSigmaRange is not None:
            fitFunc.SetParLimits(2, signalSigmaRange[0], signalSigmaRange[1])
        if signalSigma is not None:
            fitFunc.FixParameter(2, signalSigma)
        ptr = self.ptBins[bin].massPtSlice.Fit(fitFunc, "LMES0" + ("Q" if not doPrint else ""), "", fitRange[0], fitRange[1])
        # Get TFitResult object whose lifetime is controlled by Python
        fitResult.result = ptr.Clone()
        # Store the functions for plotting and calculations
        fitFuncParams = [fitFunc.GetParameters()[i] for i in range(fitFunc.GetNpar())]
        fitFuncParErrors = [fitFunc.GetParErrors()[i] for i in range(fitFunc.GetNpar())]
        nReflErr = utils.propagate_error_product(fitFuncParams[0], fitFuncParams[3], 
                                                 fitFuncParErrors[0], fitFuncParErrors[3],
                                                 fitResult.result.GetCovarianceMatrix()[0,3])
        if doPrint:
            print(f"fitFuncParams = {[str(par)+'+-'+str(err) for (par, err) in zip(fitFuncParams, fitFuncParErrors)]}")
        signalFunc = r.TF1(f"signalFunc_bin{bin}", fit_functions.signal, self.massRange[0], self.massRange[1], 3)
        signalFunc.SetParameters(np.array(fitFuncParams[0:3], dtype='d'))
        signalFunc.SetParErrors(np.array(fitFuncParErrors[0:3], dtype='d'))
        signalFunc.SetLineColor(r.kCyan+2)
        signalFunc.SetFillColor(r.kCyan-8)
        signalFunc.SetFillStyle(1001)
        signalFunc.SetNpx(1000)
        backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", backgroundFunction, self.massRange[0], self.massRange[1], backgroundNPar)
        backgroundFunc.SetParameters(np.array(fitFuncParams[3+6:3+6+backgroundNPar], dtype='d'))
        backgroundFunc.SetParErrors(np.array(fitFuncParErrors[3+6:3+6+backgroundNPar], dtype='d'))
        backgroundFunc.SetLineColor(r.kBlue)
        backgroundFunc.SetLineStyle(r.kDashed)
        backgroundFunc.SetNpx(1000)
        dataReflFunc = r.TF1(f"dataReflFunc_bin{bin}", fit_functions.reflected_background, self.massRange[0], self.massRange[1], 6)
        dataReflFunc.SetParameters(np.concatenate(([fitFuncParams[0] * fitFuncParams[3]],
                                                                    fitFuncParams[4:9]), dtype='d'))
        dataReflFunc.SetParErrors(np.concatenate(([nReflErr], fitFuncParErrors[4:9]), dtype='d'))
        dataReflFunc.SetLineColor(r.kGreen+1)
        dataReflFunc.SetLineStyle(r.kDashed)
        dataReflFunc.SetNpx(1000)
        if useCorrBkgTemplate:
            corrBkgFunc = fit_functions.make_correlated_background_model(self.massRange[0], self.massRange[1], self.ptBins[bin].histCorrBkg)
            corrBkgFunc.SetName(f"corrBkgFunc_bin{bin}")
            corrBkgFunc.SetParameter(0, fitFuncParams[3+6+backgroundNPar])
            corrBkgFunc.SetParError(0, fitFuncParErrors[3+6+backgroundNPar])
            corrBkgFunc.SetLineColor(r.kOrange+1)
            corrBkgFunc.SetLineStyle(r.kDashed)
            corrBkgFunc.SetNpx(1000)
            totalBackgroundFunc = fit_functions.make_background_model_with_corr_bkg(backgroundFunction, backgroundNPar, self.massRange[0], self.massRange[1], self.ptBins[bin].histCorrBkg)
        else:
            corrBkgFunc = None
            totalBackgroundFunc = fit_functions.make_background_model(backgroundFunction, backgroundNPar, self.massRange[0], self.massRange[1])
        totalBackgroundFunc.SetParameters(np.concatenate(([fitFuncParams[0] * fitFuncParams[3]],
                                                                    fitFuncParams[4:]), dtype='d'))
        totalBackgroundFunc.SetParErrors(np.concatenate(([nReflErr], fitFuncParErrors[4:]), dtype='d'))
        totalBackgroundFunc.SetLineColor(r.kGreen)
        totalBackgroundFunc.SetLineStyle(r.kDashed)
        totalBackgroundFunc.SetNpx(1000)
        fitResult.create_fit_results(self.massRange[0], self.massRange[1], self.ptBins[bin],
                                     fitFunc, signalFunc, backgroundFunc, dataReflFunc, corrBkgFunc, totalBackgroundFunc,
                                     doPrint=doPrint)
        # If this fit result is supposed to be the nominal one, save it to the pT bin as such
        if isNominal:
            self.ptBins[bin].nominalFitResult = fitResult
        else:
            return fitResult

    @deprecated(reason="Use fit_inv_mass instead!")
    def fit_inv_mass_old(self, bin, backgroundName, fitRange = [1.64, 2.08], initialParams = [], doPrint=True, isNominal=True):
        # Create the results object to save info on this fit
        fitResult = FitResult(fitRange[0], fitRange[1], backgroundName)
        if doPrint:
            print(f"====== Fitting bin {bin} ({self.ptBins[bin].lowerPt} < pT < {self.ptBins[bin].upperPt} GeV/c) ======")

        # Do the fit to the MC reflected background if needed
        if not hasattr(self.ptBins[bin], 'reflectedRatio'):
            self.fit_reflected(bin, doPrint)
        reflFunc = self.ptBins[bin].reflFunc
        print(f"Using reflectedRatio={self.ptBins[bin].reflectedRatio} in fit")

        # Set up fitting function
        if backgroundName == "pol2":
            fitFunc = r.TF1(f"fGaussPol2_bin{bin}", "[0]*exp(-0.5*((x-[2])/[1])^2) + ([3]*x*x + [4]*x + [5]) + (([6]*[0]*[1]) / ([8] + [9]*[11]))*(exp(-0.5*((x-[7])/[8])^2) + [9]*exp(-0.5*((x-[10])/[11])^2))", fitRange[0], fitRange[1])
            if len(initialParams) == 0:
                # Find good initial parameters
                x0 = fitRange[0]
                y0 = self.ptBins[bin].massPtSlice.GetBinContent(self.ptBins[bin].massPtSlice.GetXaxis().FindBin(x0))
                x1 = fitRange[1]
                y1 = self.ptBins[bin].massPtSlice.GetBinContent(self.ptBins[bin].massPtSlice.GetXaxis().FindBin(x1))
                b = (y1 - y0) / (x1 - x0)
                c = y0 - b*x0

                fitFunc.SetParameters(20, 0.012, 1.85, 0, b, c) 
            else:
                fitFunc.SetParameters(initialParams[0], initialParams[1], initialParams[2], initialParams[3], initialParams[4], initialParams[5])

            fitFunc.SetParNames("Amp", "Sigma", "Mean", "a", "b", "c")
            fitFunc.SetParLimits(1, 0, 0.02)
            fitFunc.SetParName(6, "reflectedRatio")
            fitFunc.FixParameter(6, self.ptBins[bin].reflectedRatio)
            fitFunc.SetParName(7, "Mean1")
            fitFunc.FixParameter(7, reflFunc.GetParameter("Mean1"))
            fitFunc.SetParName(8, "Sigma1")
            fitFunc.FixParameter(8, reflFunc.GetParameter("Sigma1"))
            fitFunc.SetParName(9, "Frac2")
            fitFunc.FixParameter(9, reflFunc.GetParameter("Amp2")/reflFunc.GetParameter("Amp1"))
            fitFunc.SetParName(10, "Mean2")
            fitFunc.FixParameter(10, reflFunc.GetParameter("Mean2"))
            fitFunc.SetParName(11, "Sigma2")
            fitFunc.FixParameter(11, reflFunc.GetParameter("Sigma2"))
        elif backgroundName == "exp":
            fitFunc = r.TF1(f"fGaussExp_bin{bin}", "[0]*exp(-0.5*((x-[2])/[1])^2) + [3]*exp([4]*x) + (([5]*[0]*[1]) / ([7] + [8]*[10]))*(exp(-0.5*((x-[6])/[7])^2) + [8]*exp(-0.5*((x-[9])/[10])^2))", fitRange[0], fitRange[1])
            if len(initialParams) == 0:
                fitFunc.SetParameters(20, 0.014, 1.85, 750, -2)
            else:
                fitFunc.SetParameters(initialParams[0], initialParams[1], initialParams[2], initialParams[3], initialParams[4])
            fitFunc.SetParNames("Amp", "Sigma", "Mean", "A", "B")
            fitFunc.SetParLimits(1, 0, 0.04)
            fitFunc.SetParName(5, "reflectedRatio")
            fitFunc.FixParameter(5, self.ptBins[bin].reflectedRatio)
            fitFunc.SetParName(6, "Mean1")
            fitFunc.FixParameter(6, reflFunc.GetParameter("Mean1"))
            fitFunc.SetParName(7, "Sigma1")
            fitFunc.FixParameter(7, reflFunc.GetParameter("Sigma1"))
            fitFunc.SetParName(8, "Frac2")
            fitFunc.FixParameter(8, reflFunc.GetParameter("Amp2")/reflFunc.GetParameter("Amp1"))
            fitFunc.SetParName(9, "Mean2")
            fitFunc.FixParameter(9, reflFunc.GetParameter("Mean2"))
            fitFunc.SetParName(10, "Sigma2")
            fitFunc.FixParameter(10, reflFunc.GetParameter("Sigma2"))
        elif backgroundName == "cheby2":
            fitFunc = r.TF1(f"fGaussCheby2_bin{bin}", "[0]*exp(-0.5*((x-[2])/[1])^2) + ([3] + [4]*x + [5]*(2*x*x - 1)) + (([6]*[0]*[1]) / ([8] + [9]*[11]))*(exp(-0.5*((x-[7])/[8])^2) + [9]*exp(-0.5*((x-[10])/[11])^2))", fitRange[0], fitRange[1])
            if len(initialParams) == 0:
                fitFunc.SetParameters(20, 0.014, 1.85, 0, 0, 0)
            else:
                fitFunc.SetParameters(initialParams[0], initialParams[1], initialParams[2], initialParams[3], initialParams[4], initialParams[5])
            fitFunc.SetParNames("Amp", "Sigma", "Mean", "A", "B", "C")
            fitFunc.SetParLimits(1, 0, 0.02)
            fitFunc.SetParName(6, "reflectedRatio")
            fitFunc.FixParameter(6, self.ptBins[bin].reflectedRatio)
            fitFunc.SetParName(7, "Mean1")
            fitFunc.FixParameter(7, reflFunc.GetParameter("Mean1"))
            fitFunc.SetParName(8, "Sigma1")
            fitFunc.FixParameter(8, reflFunc.GetParameter("Sigma1"))
            fitFunc.SetParName(9, "Frac2")
            fitFunc.FixParameter(9, reflFunc.GetParameter("Amp2")/reflFunc.GetParameter("Amp1"))
            fitFunc.SetParName(10, "Mean2")
            fitFunc.FixParameter(10, reflFunc.GetParameter("Mean2"))
            fitFunc.SetParName(11, "Sigma2")
            fitFunc.FixParameter(11, reflFunc.GetParameter("Sigma2"))
        elif backgroundName == "ratioPol2":
            fitFunc = r.TF1(f"fGaussRatioPol2_bin{bin}", "[0]*exp(-0.5*((x-[2])/[1])^2) + ([3] + [4]*x + [5]*x*x)/([6] + [7]*x + [8]*x*x) + (([9]*[0]*[1]) / ([11] + [12]*[14]))*(exp(-0.5*((x-[10])/[11])^2) + [12]*exp(-0.5*((x-[13])/[14])^2))", fitRange[0], fitRange[1])
            if len(initialParams) == 0:
                fitFunc.SetParameters(20, 0.014, 1.85, 1, 0, 0, 1, 0, 0)
            else:
                fitFunc.SetParameters(initialParams[0], initialParams[1], initialParams[2], initialParams[3], initialParams[4], initialParams[5], initialParams[6], initialParams[7], initialParams[7], initialParams[8])
            fitFunc.SetParNames("Amp", "Sigma", "Mean", "A", "B", "C", "D", "E", "F")
            fitFunc.SetParLimits(1, 0, 0.02)
            fitFunc.SetParName(9, "reflectedRatio")
            fitFunc.FixParameter(9, self.ptBins[bin].reflectedRatio)
            fitFunc.SetParName(10, "Mean1")
            fitFunc.FixParameter(10, reflFunc.GetParameter("Mean1"))
            fitFunc.SetParName(11, "Sigma1")
            fitFunc.FixParameter(11, reflFunc.GetParameter("Sigma1"))
            fitFunc.SetParName(12, "Frac2")
            fitFunc.FixParameter(12, reflFunc.GetParameter("Amp2")/reflFunc.GetParameter("Amp1"))
            fitFunc.SetParName(13, "Mean2")
            fitFunc.FixParameter(13, reflFunc.GetParameter("Mean2"))
            fitFunc.SetParName(14, "Sigma2")
            fitFunc.FixParameter(14, reflFunc.GetParameter("Sigma2"))

        else:
            raise Exception(f"Invalid background function '{backgroundName}'")

        # Fit the histogram
        if doPrint:
            print("---- Fitting data ----")
        fitResult.result = self.ptBins[bin].massPtSlice.Fit(fitFunc, "L0S" + ("Q" if not doPrint else ""), "", fitRange[0], fitRange[1])

        # Obtain the signal and background functions separately
        signalFunc = r.TF1(f"signalFunc_bin{bin}", "[0]*exp(-0.5*((x-[2])/[1])^2)", self.massRange[0], self.massRange[1])
        signalFunc.SetParameters(fitFunc.GetParameter("Amp"), fitFunc.GetParameter("Sigma"), fitFunc.GetParameter("Mean"))

        if backgroundName == "pol2":
            backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", "([0]*x*x + [1]*x + [2])", self.massRange[0], self.massRange[1])
            backgroundFunc.SetParameters(fitFunc.GetParameter("a"), fitFunc.GetParameter("b"), fitFunc.GetParameter("c"))
        elif backgroundName == "exp":
            backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", "([0]*exp([1]*x))", self.massRange[0], self.massRange[1])
            backgroundFunc.SetParameters(fitFunc.GetParameter("A"), fitFunc.GetParameter("B"))
        elif backgroundName == "cheby2":
            backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", "([0] + [1]*x + [2]*(2*x*x - 1))", self.massRange[0], self.massRange[1])
            backgroundFunc.SetParameters(fitFunc.GetParameter("A"), fitFunc.GetParameter("B"), fitFunc.GetParameter("C"))
        elif backgroundName == "ratioPol2":
            backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", "([0] + [1]*x + [2]*x*x)/([3] + [4]*x + [5]*x*x)", self.massRange[0], self.massRange[1])
            backgroundFunc.SetParameters(fitFunc.GetParameter("A"), fitFunc.GetParameter("B"), fitFunc.GetParameter("C"), fitFunc.GetParameter("D"), fitFunc.GetParameter("E"), fitFunc.GetParameter("F"))
        else:
            raise Exception(f"Background function '{backgroundName}' not implemented when extracting separate background shape!")
        backgroundFunc.SetLineColor(r.kBlue)

        dataReflFunc = r.TF1(f"dataReflFunc_bin{bin}", "[0]*(exp(-0.5*((x-[1])/[2])^2) + [3]*exp(-0.5*((x-[4])/[5])^2))", self.massRange[0], self.massRange[1])
        dataReflNorm = (fitFunc.GetParameter("reflectedRatio")*fitFunc.GetParameter("Amp")*fitFunc.GetParameter("Sigma")) / (fitFunc.GetParameter("Sigma1") + fitFunc.GetParameter("Frac2")*fitFunc.GetParameter("Sigma2"))
        dataReflFunc.SetParameters(dataReflNorm, fitFunc.GetParameter("Mean1"), fitFunc.GetParameter("Sigma1"), fitFunc.GetParameter("Frac2"), fitFunc.GetParameter("Mean2"), fitFunc.GetParameter("Sigma2"))
        dataReflFunc.SetLineColor(r.kGreen - 1)
        dataReflFunc.SetLineStyle(r.kDashed)

        totalBackgroundFunc = r.TF1(f"totalBackgroundFunc_bin{bin}",
                                                     str(backgroundFunc.GetExpFormula("p")) + "+" + str(dataReflFunc.GetExpFormula("p")),
                                                     self.massRange[0],
                                                     self.massRange[1])
        totalBackgroundFunc.SetLineColor(r.kGreen)
        totalBackgroundFunc.SetLineStyle(r.kDashed)
        fitResult.create_fit_results(self.massRange[0], self.massRange[1], self.ptBins[bin],
                                     fitFunc, signalFunc, backgroundFunc, dataReflFunc, totalBackgroundFunc,
                                     doPrint=doPrint)

        # Legacy code to calculate and overwrite signal and background counts
        fitResult.nSignal = signalFunc.Integral(fitResult.fitMu - 3*fitResult.fitSigma, fitResult.fitMu + 3*fitResult.fitSigma) / self.ptBins[bin].massPtSlice.GetBinWidth(1)
        if doPrint:
            print(f"bin {self.ptBins[bin].index} nSignal = {fitResult.nSignal} was calculated by integration from {fitResult.fitMu - 3*fitResult.fitSigma} to {fitResult.fitMu + 3*fitResult.fitSigma}")
        fitResult.nCombBackground = backgroundFunc.Integral(fitResult.fitMu - 3*fitResult.fitSigma, fitResult.fitMu + 3*fitResult.fitSigma) / self.ptBins[bin].massPtSlice.GetBinWidth(1)
        if doPrint:
            print(f"bin {self.ptBins[bin].index} nCombBackground = {fitResult.nCombBackground} was calculated by integration from {fitResult.fitMu - 3*fitResult.fitSigma} to {fitResult.fitMu + 3*fitResult.fitSigma}")
        fitResult.nReflBackground = dataReflFunc.Integral(self.massRange[0], self.massRange[1]) / self.ptBins[bin].massPtSlice.GetBinWidth(1)
        if (dataReflFunc.Eval(self.massRange[0]) > 1e-6):
            print(f"WARNING: dataReflFunc({self.massRange[0]}) = {dataReflFunc.Eval(self.massRange[0])}, normalization of reflected background may be inaccurate")
        if (dataReflFunc.Eval(self.massRange[1]) > 1e-6):
            print(f"WARNING: dataReflFunc({self.massRange[1]}) = {dataReflFunc.Eval(self.massRange[1])}, normalization of reflected background may be inaccurate")
        if doPrint:
            print(f"bin {self.ptBins[bin].index} nReflBackground = {fitResult.nReflBackground} was calculated by integration from {self.massRange[0]} to {self.massRange[1]}")
        fitResult.relativeStatError = np.sqrt(fitResult.nSignal + fitResult.nCombBackground + fitResult.nReflBackground) / fitResult.nSignal
        fitResult.signalToBackground = fitResult.nSignal / (fitResult.nCombBackground + fitResult.nReflBackground)
        fitResult.signalSignificance = fitResult.nSignal / np.sqrt(fitResult.nSignal + fitResult.nCombBackground + fitResult.nReflBackground)

        # If this fit result is supposed to be the nominal one, save it to the pT bin as such
        if isNominal:
            self.ptBins[bin].nominalFitResult = fitResult
        else:
            self.ptBins[bin].fitResults.append(fitResult)


    def calculate_efficiency_runbyrun(self, weights='lumi'):
        """
        Calculate acc x eff using MC histograms. The calculation is done separately for each run, then the weighted average is calculated using the run luminosity as the weight.
        """
        if hasattr(self, 'efficiency'):
            print("WARNING: Efficiency has already been calculated in this analysis. It will now be overwritten!")
            if self.reweighting:
                print("     WARNING: The previous efficiency was reweighted. Remember to reweight again!")

        self.runByRunEfficiencies = {}
        self.totalGenHist = r.TH1F("totalGenHist", "totalGenHist", len(self.ptBinsArray) - 1, np.asarray(self.ptBinsArray, 'd'))
        for irun, run in enumerate(self.runList):
            print(f"Calculating efficiency for run {run} ({irun+1}/{len(self.runList)})...            ", end='\r')
            with uproot.open(f'{self.dirRec}/AnalysisResults_run{run}.root') as recFile:
                recDir = recFile["analysis-asymmetric-pairing/output;1"]
            for i, item in enumerate(recDir):
                if item.member('fName') == self.groupNameD0PtMatched:
                    for ii, iitem in enumerate(recDir[i]):
                        if iitem.member('fName') == 'Pt':
                            recHistRaw = recDir[i][ii]
                            break
                    break
            recHistWritable = recHistRaw.to_writable()
            recHist = recHistWritable.to_pyroot()
            recHist = recHist.Rebin(len(self.ptBinsArray) - 1, f"PtMcMatchedFinalBins", np.asarray(self.ptBinsArray, 'd'))

            with uproot.open(f'{self.dirGen}/AnalysisResults_run{run}.root') as genFile:
                genDir = genFile["analysis-asymmetric-pairing/output;1"]
            for i, item in enumerate(genDir):
                if item.member('fName') == self.groupNameD0Generated:
                    for ii, iitem in enumerate(genDir[i]):
                        if iitem.member('fName') == 'MyMcPtYHisto':
                            genHistRaw = genDir[i][ii]
                            break
                    break
            genHistWritable = genHistRaw.to_writable()
            genHistPtY = genHistWritable.to_pyroot()

            # Project out our dy bin from the gen histogram
            lowerYBin = genHistPtY.GetYaxis().FindBin(self.minY)
            upperYBin = genHistPtY.GetYaxis().FindBin(self.maxY) - 1
            genHist = genHistPtY.ProjectionX(f"projPtMcGen_y_{self.minY}_{self.maxY}_run{run}", lowerYBin, upperYBin)
            genHist = genHist.Rebin(len(self.ptBinsArray) - 1, f"projPtMcGenFinalBins_y_{self.minY}_{self.maxY}", np.asarray(self.ptBinsArray, 'd'))

            eff = Efficiency(f'Rec. matched D0 after all cuts (run {run})', f'Gen. D0 in lumi (run {run})', recHist, genHist)
            self.runByRunEfficiencies[run] = eff

            self.totalGenHist.Add(genHist)

        print("")
        print("Complete!")
            
        # TODO: Implement weighted sum of efficiencies with lumi as weights, and propagate the error properly
        self.efficiency = r.TH1F("efficiency", f"Weighted average of efficiencies per run, weights={weights}", len(self.ptBinsArray) - 1, np.asarray(self.ptBinsArray, 'd'))
        if weights == 'mcgencount':
            # Use as weights the fraction of MCGen D0 in the run wrt. the total count. This should reproduce the "naive" calculation.
            for irun, run in enumerate(self.runList):
                weight = self.runByRunEfficiencies[run].histDenominator.GetEntries() / self.totalGenHist.GetEntries()
                self.efficiency.Add(self.runByRunEfficiencies[run].histogram, weight)
        elif weights == 'lumi':
            # Use as weights the fraction of lumi in the run wrt. the total lumi
            for irun, run in enumerate(self.runList):
                weight = self.runByRunLumi[run] / self.lumi
                self.efficiency.Add(self.runByRunEfficiencies[run].histogram, weight)

    def calculate_efficiency(self):
        if hasattr(self, 'efficiency'):
            print("WARNING: Efficiency has already been calculated in this analysis. It will now be overwritten!")
            if self.reweighting:
                print("     WARNING: The previous efficiency was reweighted. Remember to reweight again!")

        self.efficiencyFine = self.histD0PtMatched.Clone()
        self.efficiencyFine.Divide(self.histD0PtGenerated)
        print(f"Rec/Gen histogram was calculated using '{self.groupNameD0PtMatched}' and '{self.groupNameD0Generated}'")
        self.efficiencyFine.SetName("efficiencyFine")
        self.efficiencyFine.SetTitle("Reconstructed, matched D0 / Generated D0 after BC cuts")
        if self.rapidityGapSelectionEfficiency is not None:
            print(f"INFO: Multiplying 'efficiencyFine' by rapidityGapSelectionEfficiency={self.rapidityGapSelectionEfficiency}")
            self.efficiencyFine.Scale(self.rapidityGapSelectionEfficiency)
            self.efficiencyFine.SetTitle("(" + self.efficiencyFine.GetTitle() + ") #times #epsilon(rap. gap)")
        # Make sure the pT axis is (0, 12) GeV
        if (self.efficiencyFine.FindBin(self.ptBinsArray[-1]) <= self.efficiencyFine.GetNbinsX()):
            nFineBins = self.efficiencyFine.FindBin(self.ptBinsArray[-1])
            histTemp = r.TH1F("efficiencyFine", "Reconstructed, matched D0 / Generated D0 after BC cuts", nFineBins, self.ptBinsArray[0], self.ptBinsArray[-1])
            for i in range(1, nFineBins + 1):
                histTemp.SetBinContent(i, self.efficiencyFine.GetBinContent(i))
                histTemp.SetBinError(i, self.efficiencyFine.GetBinError(i))

            self.efficiencyFine = histTemp
        # Create a new version of the correction factor hist with the final binning and range
        self.histD0PtMatchedFinalBins = self.histD0PtMatched.Rebin(len(self.ptBinsArray) - 1, f"PtMcMatchedFinalBins", np.asarray(self.ptBinsArray, 'd'))
        self.histD0PtGeneratedFinalBins = self.histD0PtGenerated.Rebin(len(self.ptBinsArray) - 1, f"projPtMcGenFinalBins_y_{self.minY}_{self.maxY}", np.asarray(self.ptBinsArray, 'd'))
        self.efficiency = self.histD0PtMatchedFinalBins.Clone()
        self.efficiency.SetName("efficiency")
        self.efficiency.SetTitle("Reconstructed / Generated;p_{T};Ratio")
        self.efficiency.Divide(self.histD0PtGeneratedFinalBins)
        if self.rapidityGapSelectionEfficiency is not None:
            print(f"INFO: Multiplying 'efficiency' by rapidityGapSelectionEfficiency={self.rapidityGapSelectionEfficiency}")
            self.efficiency.Scale(self.rapidityGapSelectionEfficiency)
            self.efficiency.SetTitle("(" + self.efficiency.GetTitle() + ") #times #epsilon(rap. gap)")

    def calculate_corrected_spectrum(self):
        if hasattr(self, 'histCorrectedSpectrum'):
            print("WARNING: The corrected spectrum was calculated for this analysis before!")
        # Apply correction factor
        self.histCorrectedSpectrum = self.histRawYield.Clone()
        self.histCorrectedSpectrum.SetName("histCorrectedSpectrum")
        self.histCorrectedSpectrum.SetTitle("Corrected pT spectrum")
        self.histCorrectedSpectrum.GetYaxis().SetTitle("dN/dp_{T}")
        self.histCorrectedSpectrum.Divide(self.efficiency)

    def reweight_efficiency(self):
        if self.reweighting:
            raise Exception("Reweighting was already done for this analysis!")
        self.reweighting = True
        if self.histCorrectedSpectrum is None:
            print("Corrected spectrum was not calculated yet, doing it now.")
            self.calculate_corrected_spectrum()

        self.reweightingFunc = r.TF1("powerLaw", "[0]*x/TMath::Power((1+TMath::Power(x/[1],[3])),[2])", 0, 12)
        self.reweightingFunc.SetParameters(394000, 1.83, 1.78, 2.87)
        print("===== Fitting to the corrected spectrum for reweighting =====")
        self.reweightingFitResults = self.histCorrectedSpectrum.Fit(self.reweightingFunc, "LS0")
        self.efficiencyReweighted = self.efficiency.Clone()
        self.efficiencyReweighted.Reset()
        self.efficiencyReweighted.SetName("efficiencyReweighted")
        self.efficiencyReweighted.SetTitle("(Reconstructed / Generated) " + ("#times #epsilon(rap. gap) " if self.rapidityGapSelectionEfficiency is not None else "") + "(reweighted)")
        dpT = self.efficiencyFine.GetBinWidth(1)
        # Calculate numerator of <epsilon>_i
        for i in range(1, self.efficiencyFine.GetNbinsX() + 1):
            binCenter = self.efficiencyFine.GetBinCenter(i)
            binContent = self.efficiencyFine.GetBinContent(i)
            binError = self.efficiencyFine.GetBinError(i)
            weight = self.reweightingFunc.Eval(binCenter)
            # Calculate weighted content
            weightedContent = binContent * weight * dpT
            weightedError = np.abs(weight) * dpT * binError # Error propagation
            # Find the target bin and fill it
            targetBin = self.efficiencyReweighted.FindBin(binCenter)
            self.efficiencyReweighted.AddBinContent(targetBin, weightedContent)
            currentErrorTarget = self.efficiencyReweighted.GetBinError(targetBin)
            self.efficiencyReweighted.SetBinError(targetBin, np.sqrt(currentErrorTarget**2 + weightedError**2))
        # Calculate denominator of <epsilon>_i
        histDenominator = self.efficiency.Clone()
        histDenominator.Reset()
        # For some reason the first iteration of IntegralError gives a nonsensical value, throw this away first
        _ = self.reweightingFunc.IntegralError(0., 0.1, self.reweightingFitResults.GetParams(), self.reweightingFitResults.GetCovarianceMatrix().GetMatrixArray(), epsilon=1e-8)
        for i in range(1, self.efficiencyReweighted.GetNbinsX() + 1):
            lowEdge = self.efficiencyReweighted.GetXaxis().GetBinLowEdge(i)
            upEdge = self.efficiencyReweighted.GetXaxis().GetBinUpEdge(i)
            print(f"--- Calculating integral of reweightingFunc from {lowEdge} to {upEdge} ---")
            integral = self.reweightingFunc.Integral(lowEdge, upEdge)
            histDenominator.SetBinContent(i, integral)
            integralError = self.reweightingFunc.IntegralError(lowEdge, upEdge, self.reweightingFitResults.GetParams(), self.reweightingFitResults.GetCovarianceMatrix().GetMatrixArray(), epsilon=1e-8)
            print(f"Bin {i}: integral = {integral} +- {integralError}")
            histDenominator.SetBinError(i, integralError)
        # Obtain <epsilon>_i as a histogram
        self.efficiencyReweighted.Divide(histDenominator)
        self.efficiencyWithoutReweighting = self.efficiency
        self.efficiencyWithoutReweighting.SetLineColor(r.kRed)
        self.efficiencyWithoutReweighting.SetTitle("(Reconstructed / Generated) " + "#times #epsilon(rap. gap) " if self.rapidityGapSelectionEfficiency is not None else "" + "(without reweighting)")
        self.efficiency = self.efficiencyReweighted

        # Apply reweighted correction factor
        self.histCorrectedSpectrumReweighted = self.histRawYield.Clone()
        self.histCorrectedSpectrumReweighted.SetName("histCorrectedSpectrumWithReweighting")
        self.histCorrectedSpectrumReweighted.SetTitle("Corrected pT spectrum with reweighting")
        self.histCorrectedSpectrumReweighted.GetYaxis().SetTitle("dN/dp_{T}")
        self.histCorrectedSpectrumReweighted.Divide(self.efficiency)
        self.histCorrectedSpectrumWithoutReweighting = self.histCorrectedSpectrum
        self.histCorrectedSpectrumWithoutReweighting.SetLineColor(r.kRed)
        self.histCorrectedSpectrumWithoutReweighting.SetTitle("Corrected spectrum, without reweighting")
        self.histCorrectedSpectrum = self.histCorrectedSpectrumReweighted

    def calculate_track_cut_efficiencies(self, directory, **kwargs):
        """
        Calculate partial efficiencies for track cuts
        """
        trackCutNames = {
            "kaonTPCCut"    : "D0KaonTPC",
            "kaonTOFCut"    : "D0KaonTOF",
            "kaonFullCut"   : "D0KaonTPCTOF",
            "pionCut"       : "noTrackCut",
            "etaCut"        : "D0CommonTrackCutsEta",
            "ptCut"         : "D0CommonTrackCutsPt",
            "itsQualityCut" : "D0CommonTrackCutsITSib",
            "tpcNClsCut"    : "D0CommonTrackCutsTPCncls",
            "tpcChi2Cut"    : "D0CommonTrackCutsTPCchi2",
            "dcaZCut"       : "D0CommonTrackCutsDCAz",
            "fullCommonCut" : "D0CommonTrackCuts"
        }
        trackCutNames.update(kwargs)

        r.TH1.AddDirectory(r.kFALSE)
        self.trackCutEfficiencies = []
        fullNames = ["noTrackCut:noTrackCut", 
                     f"{trackCutNames['kaonTPCCut']}:noTrackCut", f"{trackCutNames['kaonTOFCut']}:noTrackCut", 
                     f"noTrackCut:noTrackCut_{trackCutNames['etaCut']}", f"noTrackCut:noTrackCut_{trackCutNames['ptCut']}",
                     f"noTrackCut:noTrackCut_{trackCutNames['itsQualityCut']}", f"noTrackCut:noTrackCut_{trackCutNames['tpcNClsCut']}",
                     f"noTrackCut:noTrackCut_{trackCutNames['dcaZCut']}", f"noTrackCut:noTrackCut_{trackCutNames['tpcChi2Cut']}",
                     f"{trackCutNames['kaonFullCut']}:noTrackCut_{trackCutNames['fullCommonCut']}"]
        fullNames = ["analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_" + fullName + "_KPiFromD0/Y_PtFine" for fullName in fullNames]
        tmpDict = get_histograms(directory, self.runList, fullNames)
        # Reconstructed, matched D0 in selected events, but with no track or pair cuts. This is the denominator for all partial efficiencies due to track cuts
        histDenom2D = tmpDict["analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_KPiFromD0/Y_PtFine"]
        histDenom = project_fiducial_acceptance(histDenom, self.minY, self.maxY)
        histDenom = histDenom.Rebin(len(self.ptBinsArray) - 1, histDenom.GetName(), np.asarray(self.ptBinsArray, 'd'))

        histD0PtKaonTPC2D = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{trackCutNames['kaonTPCCut']}:noTrackCut_KPiFromD0/Y_PtFine"]
        histD0PtKaonTPC = project_fiducial_acceptance(histD0PtKaonTPC, self.minY, self.maxY)
        histD0PtKaonTPC = histD0PtKaonTPC.Rebin(len(self.ptBinsArray) - 1, histD0PtKaonTPC.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., kaon TPC nSigma<3", "Rec. matched D0 in sel evt.", histD0PtKaonTPC, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtKaonTOF2D = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{trackCutNames['kaonTOFCut']}:noTrackCut_KPiFromD0/Y_PtFine"]
        histD0PtKaonTOF = project_fiducial_acceptance(histD0PtKaonTOF, self.minY, self.maxY)
        histD0PtKaonTOF = histD0PtKaonTOF.Rebin(len(self.ptBinsArray) - 1, histD0PtKaonTOF.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., kaon TOF nSigma<3", "Rec. matched D0 in sel evt.", histD0PtKaonTOF, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtEtaCut2D = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['etaCut']}_KPiFromD0/Y_PtFine"]
        histD0PtEtaCut = project_fiducial_acceptance(histD0PtEtaCut, self.minY, self.maxY)
        histD0PtEtaCut = histD0PtEtaCut.Rebin(len(self.ptBinsArray) - 1, histD0PtEtaCut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track |eta|<0.9", "Rec. matched D0 in sel evt.", histD0PtEtaCut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtPtCut2D = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['ptCut']}_KPiFromD0/Y_PtFine"]
        histD0PtPtCut = project_fiducial_acceptance(histD0PtPtCut, self.minY, self.maxY)
        histD0PtPtCut = histD0PtPtCut.Rebin(len(self.ptBinsArray) - 1, histD0PtPtCut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track pT>0.5 GeV/c", "Rec. matched D0 in sel evt.", histD0PtPtCut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtITSibCut2D = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['itsQualityCut']}_KPiFromD0/Y_PtFine"]
        histD0PtITSibCut = project_fiducial_acceptance(histD0PtITSibCut, self.minY, self.maxY)
        histD0PtITSibCut = histD0PtITSibCut.Rebin(len(self.ptBinsArray) - 1, histD0PtITSibCut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track IsITSibAny=1", "Rec. matched D0 in sel evt.", histD0PtITSibCut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtTPCnclsCut2D = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['tpcNClsCut']}_KPiFromD0/Y_PtFine"]
        histD0PtTPCnclsCut = project_fiducial_acceptance(histD0PtTPCnclsCut, self.minY, self.maxY)
        histD0PtTPCnclsCut = histD0PtTPCnclsCut.Rebin(len(self.ptBinsArray) - 1, histD0PtTPCnclsCut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track TPC nCls > 50", "Rec. matched D0 in sel evt.", histD0PtTPCnclsCut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtDCAzCut2D = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['dcaZCut']}_KPiFromD0/Y_PtFine"]
        histD0PtDCAzCut = project_fiducial_acceptance(histD0PtDCAzCut, self.minY, self.maxY)
        histD0PtDCAzCut = histD0PtDCAzCut.Rebin(len(self.ptBinsArray) - 1, histD0PtDCAzCut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track |DCAz| < 0.3 cm", "Rec. matched D0 in sel evt.", histD0PtDCAzCut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtTPCchi2Cut2D = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['tpcChi2Cut']}_KPiFromD0/Y_PtFine"]
        histD0PtTPCchi2Cut = project_fiducial_acceptance(histD0PtTPCchi2Cut, self.minY, self.maxY)
        histD0PtTPCchi2Cut = histD0PtTPCchi2Cut.Rebin(len(self.ptBinsArray) - 1, histD0PtTPCchi2Cut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track TPCchi2 < 4", "Rec. matched D0 in sel evt.", histD0PtTPCchi2Cut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtAllTrackCuts2D = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{trackCutNames['kaonFullCut']}:noTrackCut_{trackCutNames['fullCommonCut']}_KPiFromD0/Y_PtFine"]
        histD0PtAllTrackCuts = project_fiducial_acceptance(histD0PtAllTrackCuts, self.minY, self.maxY)
        histD0PtAllTrackCuts = histD0PtAllTrackCuts.Rebin(len(self.ptBinsArray) - 1, histD0PtAllTrackCuts.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., after all track cuts", "Rec. matched D0 in sel evt.", histD0PtAllTrackCuts, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff

    def calculate_factorized_efficiencies(self, allowOneDimHistograms=False):
        """
        Calculate factorized efficiencies for some predefined (hardcoded) factorizations
        Total efficiency = N(rec. matched D0 after all cuts) / N(gen. D0 after BC cuts)
        If allowOneDimHistograms is True, it is assumed that the fiducial acceptance cut was applied inside table-reader,
        so 1-dimensional pT histograms are safe for efficiency calculations.
        """

        # Prepare histograms
        if not hasattr(self, 'histD0PtGeneratedFinalBins'):
            self.histD0PtGeneratedFinalBins = self.histD0PtGenerated.Rebin(len(self.ptBinsArray) - 1, f"projPtMcGenFinalBins_y_{self.minY}_{self.maxY}", np.asarray(self.ptBinsArray, 'd'))

        self.histD0PtYGenInRecEvent = self.dictGenHists[f"analysis-asymmetric-pairing/output;1/MCTruthGenRec_D0FS/PtMC_YMC"]
        lowerYBin = self.histD0PtYGenInRecEvent.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenInRecEvent.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenInRecEvent = self.histD0PtYGenInRecEvent.ProjectionX(f"projPtMcGenInRecEvent_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenInRecEvent.SetTitle(f"Generated D0 in reconstructed event, {self.minY} < y < {self.maxY}")
        self.histD0PtGenInRecEvent = self.histD0PtGenInRecEvent.Rebin(len(self.ptBinsArray) - 1, f"PtMcGenInRecEventFinalBins", np.asarray(self.ptBinsArray, 'd'))

        self.histD0PtYGenInSelEvent = self.dictGenHists[f"analysis-asymmetric-pairing/output;1/MCTruthGenSel_D0FS/PtMC_YMC"]
        lowerYBin = self.histD0PtYGenInSelEvent.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenInSelEvent.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenInSelEvent = self.histD0PtYGenInSelEvent.ProjectionX(f"projPtMcGenInSelEvent_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenInSelEvent.SetTitle(f"Generated D0 in selected event, {self.minY} < y < {self.maxY}")
        self.histD0PtGenInSelEvent = self.histD0PtGenInSelEvent.Rebin(len(self.ptBinsArray) - 1, f"PtMcGenInSelEventFinalBins", np.asarray(self.ptBinsArray, 'd'))

        self.histD0PtYGenInSelEventDaughtersInAcc = self.dictGenHists[f"analysis-asymmetric-pairing/output;1/MCTruthGenSelDaughtersInAcc_KPiFromD0FS/PtMC_YMC"]
        lowerYBin = self.histD0PtYGenInSelEventDaughtersInAcc.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenInSelEventDaughtersInAcc.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenInSelEventDaughtersInAcc = self.histD0PtYGenInSelEventDaughtersInAcc.ProjectionX(f"projPtMcGenInSelEventDaughtersInAcc_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenInSelEventDaughtersInAcc.SetTitle(f"Generated D0 in selected event with both daughters in acceptance, {self.minY} < y < {self.maxY}")
        self.histD0PtGenInSelEventDaughtersInAcc = self.histD0PtGenInSelEventDaughtersInAcc.Rebin(len(self.ptBinsArray) - 1, f"PtMcGenInSelEventDaughtersInAccFinalBins", np.asarray(self.ptBinsArray, 'd'))

        self.histD0PtYGenInRecEventDaughtersInAcc = self.dictGenHists[f"analysis-asymmetric-pairing/output;1/MCTruthGenRecDaughtersInAcc_KPiFromD0FS/PtMC_YMC"]
        lowerYBin = self.histD0PtYGenInRecEventDaughtersInAcc.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenInRecEventDaughtersInAcc.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenInRecEventDaughtersInAcc = self.histD0PtYGenInRecEventDaughtersInAcc.ProjectionX(f"projPtMcGenInRecEventDaughtersInAcc_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenInRecEventDaughtersInAcc.SetTitle(f"Generated D0 in selected event with both daughters in acceptance, {self.minY} < y < {self.maxY}")
        self.histD0PtGenInRecEventDaughtersInAcc = self.histD0PtGenInRecEventDaughtersInAcc.Rebin(len(self.ptBinsArray) - 1, f"PtMcGenInRecEventDaughtersInAccFinalBins", np.asarray(self.ptBinsArray, 'd'))

        self.histD0PtYGenAfterBcCutsDaughtersInAcc = self.dictGenHists[f"analysis-asymmetric-pairing/output;1/MCTruthGenAfterBcCutsDaughtersInAcc_KPiFromD0FS/PtMC_YMC"]
        lowerYBin = self.histD0PtYGenAfterBcCutsDaughtersInAcc.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenAfterBcCutsDaughtersInAcc.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenAfterBcCutsDaughtersInAcc = self.histD0PtYGenAfterBcCutsDaughtersInAcc.ProjectionX(f"projPtMcGenAfterBcCutsDaughtersInAcc_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenAfterBcCutsDaughtersInAcc.SetTitle(f"Generated D0 in selected event with both daughters in acceptance, {self.minY} < y < {self.maxY}")
        self.histD0PtGenAfterBcCutsDaughtersInAcc = self.histD0PtGenAfterBcCutsDaughtersInAcc.Rebin(len(self.ptBinsArray) - 1, f"PtMcGenAfterBcCutsDaughtersInAccFinalBins", np.asarray(self.ptBinsArray, 'd'))

        self.histD0PtMatchedInRecEvent = self.get_reconstructed_mc_histogram("analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_beforeEvtCuts_noTrackCut:noTrackCut_KPiFromD0FS",
                                                                             "PtMcMatchedInRecEvent",
                                                                             "Reconstructed, matched D0->Kpi in reconstructed event, no track or pair cuts")

        self.histD0PtMatchedInSelEvent = self.get_reconstructed_mc_histogram("analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_KPiFromD0FS",
                                                                             "PtMcMatchedInSelEvent",
                                                                             "Reconstructed, matched D0->Kpi in selected event, no track or pair cuts")

        self.histD0PtMatchedInSelEventKaonTPCPID = self.get_reconstructed_mc_histogram("analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_D0KaonTPC:noTrackCut_KPiFromD0FS",
                                                                                       "PtMcMatchedInSelEventKaonTPCPID",
                                                                                       "Reconstructed, matched D0->Kpi in selected event, kaon passed TPC PID")

        self.histD0PtMatchedInSelEventKaonTPCTOFPID = self.get_reconstructed_mc_histogram("analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_D0KaonTPCTOF:noTrackCut_KPiFromD0FS",
                                                                                          "PtMcMatchedInSelEventKaonTPCTOFPID",
                                                                                          "Reconstructed, matched D0->Kpi in selected event, kaon passed TPC+TOF PID")

        self.histD0PtMatchedInSelEventAfterTrackCuts = self.get_reconstructed_mc_histogram(f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4_KPiFromD0FS",
                                                                                           "PtMcMatchedInSelEventAfterTrackCuts",
                                                                                           "Reconstructed, matched D0->Kpi in selected event, selected tracks, no pair cuts")

        if not hasattr(self, 'histD0PtMatchedFinalBins'):
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts = self.get_reconstructed_mc_histogram(f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4{self.pairCutName}_KPiFromD0FS",  "PtMcMatchedInSelEventAfterTrackCutsAndPairCuts", "Reconstructed, matched D0->Kpi in selected event, selected tracks, selected pairs")
        else:
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts = self.histD0PtMatchedFinalBins.Clone()
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts.SetName("PtMcMatchedInSelEventAfterTrackCutsAndPairCuts")
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts.SetTitle("Reconstructed, matched D0->Kpi in selected event, selected tracks, selected pairs")

        # List to hold factorized efficiencies
        self.factorizedEfficiencies = []

        # Calculate efficiencies
        eff = FactorizedEfficiency(2)
        eff.add_factor("Reconstructed, matched D0 after all cuts", "MC gen D0 in reconstructed events", self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts, self.histD0PtGenInRecEvent)
        eff.add_factor("MC gen D0 in reconstructed events", "MC gen D0 in all events passing BC cuts", self.histD0PtGenInRecEvent, self.histD0PtGeneratedFinalBins)
        eff.calculate_total_efficiency()
        self.factorizedEfficiencies.append(eff)
        del eff

        eff = FactorizedEfficiency(6)
        eff.add_factor("Gen. D0 in lumi w/ accepted daughters", "Gen. D0 in lumi", self.histD0PtGenAfterBcCutsDaughtersInAcc, self.histD0PtGeneratedFinalBins)
        eff.add_factor("Gen. D0 in rec. evt. w/ accepted daughters", "Gen. D0 in lumi w/ accepted daughters", self.histD0PtGenInRecEventDaughtersInAcc, self.histD0PtGenAfterBcCutsDaughtersInAcc)
        eff.add_factor("Gen. D0 in sel. evt. w/ accepted daughters", "Gen. D0 in rec. evt. w/ accepted daughters", self.histD0PtGenInSelEventDaughtersInAcc, self.histD0PtGenInRecEventDaughtersInAcc)
        eff.add_factor("Rec. matched D0 in sel. evt", "Gen. D0 in sel. evt. w/ accepted daughters", self.histD0PtMatchedInSelEvent, self.histD0PtGenInSelEventDaughtersInAcc)
        eff.add_factor("Rec. matched D0 in sel. evt. after track cuts", "Rec. matched D0 in sel. evt.", self.histD0PtMatchedInSelEventAfterTrackCuts, self.histD0PtMatchedInSelEvent)
        eff.add_factor("Rec. matched D0 in sel. evt. after track and pair cuts", "Rec. matched D0 in sel. evt. after track cuts", self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts, self.histD0PtMatchedInSelEventAfterTrackCuts)
        eff.calculate_total_efficiency()
        self.factorizedEfficiencies.append(eff)
        del eff

        # This factorization may not be possible with 'legacy' MC AnalysisResults
        if self.histD0PtMatchedInRecEvent is not None and self.histD0PtMatchedInSelEventKaonTPCPID is not None and self.histD0PtMatchedInSelEventKaonTPCTOFPID is not None:
            eff = FactorizedEfficiency(6)
            eff.add_factor("Gen. D0 in lumi w/ accepted daughters", "Gen. D0 in lumi", self.histD0PtGenAfterBcCutsDaughtersInAcc, self.histD0PtGeneratedFinalBins)
            eff.add_factor("Rec. matched D0 in rec. evt.", "Gen. D0 in lumi w/ accepted daughters", self.histD0PtMatchedInRecEvent, self.histD0PtGenAfterBcCutsDaughtersInAcc)
            eff.add_factor("Rec. matched D0 in sel. evt.", "Rec. matched D0 in rec. evt.", self.histD0PtMatchedInSelEvent, self.histD0PtMatchedInRecEvent)
            eff.add_factor("Rec. matched D0 in sel. evt., kaon passed TPC PID", "Rec. matched D0 in sel. evt.", self.histD0PtMatchedInSelEventKaonTPCPID, self.histD0PtMatchedInSelEvent)
            eff.add_factor("Rec. matched D0 in sel. evt., kaon passed TPC+TOF PID", "Rec. matched D0 in sel. evt., kaon passed TPC PID", self.histD0PtMatchedInSelEventKaonTPCTOFPID, self.histD0PtMatchedInSelEventKaonTPCPID)
            eff.add_factor("Rec. matched D0 in sel. evt., all cuts passed", "Rec. matched D0 in sel. evt., kaon passed TPC+TOF PID", self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts, self.histD0PtMatchedInSelEventKaonTPCTOFPID)
            eff.calculate_total_efficiency()
            self.factorizedEfficiencies.append(eff)
            del eff

        # This factorization may not be possible with 'legacy' MC AnalysisResults
        if self.histD0PtMatchedInRecEvent is not None and self.histD0PtMatchedInSelEventKaonTPCPID is not None and self.histD0PtMatchedInSelEventKaonTPCTOFPID is not None:
            eff = FactorizedEfficiency(7)
            eff.add_factor("Gen. D0 in lumi w/ accepted daughters", "Gen. D0 in lumi", self.histD0PtGenAfterBcCutsDaughtersInAcc, self.histD0PtGeneratedFinalBins)
            eff.add_factor("Gen. D0 in rec. evt. w/ accepted daughters", "Gen. D0 in lumi w/ accepted daughters", self.histD0PtGenInRecEventDaughtersInAcc, self.histD0PtGenAfterBcCutsDaughtersInAcc)
            eff.add_factor("Rec. matched D0 in rec. evt.", "Gen. D0 in rec. evt. w/ accepted daughters", self.histD0PtMatchedInRecEvent, self.histD0PtGenInRecEventDaughtersInAcc)
            eff.add_factor("Rec. matched D0 in sel. evt.", "Rec. matched D0 in rec. evt.", self.histD0PtMatchedInSelEvent, self.histD0PtMatchedInRecEvent)
            eff.add_factor("Rec. matched D0 in sel. evt., kaon passed TPC PID", "Rec. matched D0 in sel. evt.", self.histD0PtMatchedInSelEventKaonTPCPID, self.histD0PtMatchedInSelEvent)
            eff.add_factor("Rec. matched D0 in sel. evt., kaon passed TPC+TOF PID", "Rec. matched D0 in sel. evt., kaon passed TPC PID", self.histD0PtMatchedInSelEventKaonTPCTOFPID, self.histD0PtMatchedInSelEventKaonTPCPID)
            eff.add_factor("Rec. matched D0 in sel. evt., all cuts passed", "Rec. matched D0 in sel. evt., kaon passed TPC+TOF PID", self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts, self.histD0PtMatchedInSelEventKaonTPCTOFPID)
            eff.calculate_total_efficiency()
            self.factorizedEfficiencies.append(eff)
            del eff

        # This factorization may not be possible with 'legacy' MC AnalysisResults
        if self.histD0PtMatchedInRecEvent is not None and self.histD0PtMatchedInSelEventKaonTPCPID is not None and self.histD0PtMatchedInSelEventKaonTPCTOFPID is not None:
            eff = FactorizedEfficiency(8)
            eff.add_factor("Gen. D0 in lumi w/ accepted daughters", "Gen. D0 in lumi", self.histD0PtGenAfterBcCutsDaughtersInAcc, self.histD0PtGeneratedFinalBins)
            eff.add_factor("Gen. D0 in rec. evt. w/ accepted daughters", "Gen. D0 in lumi w/ accepted daughters", self.histD0PtGenInRecEventDaughtersInAcc, self.histD0PtGenAfterBcCutsDaughtersInAcc)
            eff.add_factor("Rec. matched D0 in rec. evt.", "Gen. D0 in rec. evt. w/ accepted daughters", self.histD0PtMatchedInRecEvent, self.histD0PtGenInRecEventDaughtersInAcc)
            eff.add_factor("Rec. matched D0 in sel. evt.", "Rec. matched D0 in rec. evt.", self.histD0PtMatchedInSelEvent, self.histD0PtMatchedInRecEvent)
            eff.add_factor("Rec. matched D0 in sel. evt., kaon passed TPC PID", "Rec. matched D0 in sel. evt.", self.histD0PtMatchedInSelEventKaonTPCPID, self.histD0PtMatchedInSelEvent)
            eff.add_factor("Rec. matched D0 in sel. evt., kaon passed TPC+TOF PID", "Rec. matched D0 in sel. evt., kaon passed TPC PID", self.histD0PtMatchedInSelEventKaonTPCTOFPID, self.histD0PtMatchedInSelEventKaonTPCPID)
            eff.add_factor("Rec. matched D0 in sel. evt., all track cuts passed", "Rec. matched D0 in sel. evt., kaon passed TPC+TOF PID", self.histD0PtMatchedInSelEventAfterTrackCuts, self.histD0PtMatchedInSelEventKaonTPCTOFPID)
            eff.add_factor("Rec. matched D0 in sel. evt., track and pair cuts passed", "Rec. matched D0 in sel. evt., all track cuts passed", self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts, self.histD0PtMatchedInSelEventAfterTrackCuts)
            eff.calculate_total_efficiency()
            self.factorizedEfficiencies.append(eff)
            del eff

    def get_reconstructed_mc_histogram(self, groupName, name, title):
        if f"{groupName}/Y_PtFine" in self.dictRecHists:
            histD0YPt = self.dictRecHists[f"{groupName}/Y_PtFine"]
            histD0Pt = project_fiducial_acceptance(histD0YPt, self.minY, self.maxY)
        elif f"{groupName}/Pt" in self.dictRecHists:
            print("Using 1-dimensional pT histogram for reconstructed count! For this to be correct, the fiducial rapidity cut should have been applied upstream!")
            histD0Pt = self.dictRecHists[f"{groupName}/Pt"]
        else:
            print(f"Could not find histogram for '{title}' ({name})! Tried {groupName}/Pt and {groupName}/Y_PtFine")
            return None
        histD0Pt = histD0Pt.Rebin(len(self.ptBinsArray) - 1, name, np.asarray(self.ptBinsArray, 'd'))
        histD0Pt.SetTitle(title)
        return histD0Pt

    def create_raw_yield_histogram(self):
        self.histRawYield = r.TH1F("histRawYield", "Raw yield /#Delta p_{T}, raw stat. errors", len(self.ptBins), np.asarray(self.ptBinsArray, 'd'))
        self.histRawYield.GetYaxis().SetTitle("Raw D^{0} yield (1/GeV c^{-1})")
        self.histRawYield.GetXaxis().SetTitle("p_{T} (GeV c^{-1})")
        for i, bin in enumerate(self.ptBins):
            self.histRawYield.SetBinContent(i+1, bin.nominalFitResult.nSignal / self.histRawYield.GetBinWidth(i+1))
            # Error propagation with the bin width
            # NOTE: The error used here is the symmetric Hesse error! For a final result, the asymmetric Minos error should be used, but that doesn't work with TH1 
            self.histRawYield.SetBinError(i+1, bin.nominalFitResult.relativeStatError / self.histRawYield.GetBinWidth(i+1) * bin.nominalFitResult.nSignal)
        self.histRawYield.SetStats(0)

    def calculate_spectrum_per_event(self, draw=True):
        self.histCorrectedSpectrumPerEvent = self.histCorrectedSpectrum.Clone()
        self.histCorrectedSpectrumPerEvent.SetName("histCorrectedSpectrumPerEvent")
        self.histCorrectedSpectrumPerEvent.SetTitle("Corrected pT spectrum normalized by number of events")
        self.histCorrectedSpectrumPerEvent.Scale(1. / (self.maxY - self.minY)) # Divide by dy
        self.histCorrectedSpectrumPerEvent.Scale(1. / self.nEvents) # Normalize by number of events
        self.histCorrectedSpectrumPerEvent.GetYaxis().SetTitle("1/N_{events} d^{2}N/dp_{T}dy")
        if draw:
            if hasattr(self, 'canvasPtSpectrumPerEvent'):
                del self.canvasPtSpectrumPerEvent
            self.canvasPtSpectrumPerEvent = r.TCanvas("canvasPtSpectrumPerEvent")
            self.canvasPtSpectrumPerEvent.cd()
            self.histCorrectedSpectrumPerEvent.Draw()
            r.gPad.SetLogy()
            self.canvasPtSpectrumPerEvent.Draw()

    def calculate_raw_yield_per_lumi(self, draw=True):
        self.histRawYieldPerLumi = self.histRawYield.Clone()
        self.histRawYieldPerLumi.SetName("histRawYieldPerLumi")
        self.histRawYieldPerLumi.SetTitle("Raw yield /#Delta p_{T} L_{int}")
        self.histRawYieldPerLumi.Scale(1 / self.lumi) # µb / GeVc^-1
        self.histRawYieldPerLumi.Scale(1 / 1000.) # mb / GeVc^-1
        self.histRawYieldPerLumi.GetYaxis().SetTitle("Raw D^{0} yield / L_{int} (mb/GeV c^{-1})")
        self.histRawYieldPerLumi.SetStats(0)
        if draw:
            if hasattr(self, 'canvasRawYieldPerLumi'):
                del self.canvasRawYieldPerLumi
            self.canvasRawYieldPerLumi = r.TCanvas("canvasRawYieldPerLumi")
            self.canvasRawYieldPerLumi.cd()
            self.histRawYieldPerLumi.Draw()
            r.gPad.SetLogy()
            self.canvasRawYieldPerLumi.Draw()

    def calculate_raw_yield_per_event(self, draw=True):
        self.histRawYieldPerEvent = self.histRawYield.Clone()
        self.histRawYieldPerEvent.SetName("histRawYieldPerEvent")
        self.histRawYieldPerEvent.SetTitle("Raw yield /#Delta p_{T} N vis. evt.")
        self.histRawYieldPerEvent.Scale(1 / self.nEvents)
        self.histRawYieldPerEvent.GetYaxis().SetTitle("Raw D^{0} yield / N vis. evt. (1/GeV c^{-1})")
        self.histRawYieldPerEvent.SetStats(0)
        if draw:
            if hasattr(self, 'canvasRawYieldPerEvent'):
                del self.canvasRawYieldPerEvent
            self.canvasRawYieldPerEvent = r.TCanvas("canvasRawYieldPerEvent")
            self.canvasRawYieldPerEvent.cd()
            self.histRawYieldPerEvent.Draw()
            r.gPad.SetLogy()
            self.canvasRawYieldPerEvent.Draw()

    def rescale_ir_dependence(self, theta, cov_theta, n_data=2):
        """
        Apply data-driven IR dependence using fit parameters in theta, and the covariance matrix cov_theta.
        Replaces the efficiency and the corrected spectrum
        """
        # Get the histograms for IR dependence rescaling
        hist_rec_3d = self.dictMcHists[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4{self.pairCutName}_KPiFromD0FS/MyYPtInteractionRateHisto"]
        hist_rec_3d.GetXaxis().SetRangeUser(self.minY, self.maxY)
        hist_rec_2d = hist_rec_3d.Project3D('zy')
        hist_rec_2d.SetName("hist_rec_2d")

        hist_gen_3d = self.dictGenHists["analysis-asymmetric-pairing/output;1/MCTruthGenAfterBcCuts_D0FS/MyMcPtYInteractionRateHisto"]
        hist_gen_3d.GetYaxis().SetRangeUser(self.minY, self.maxY)
        hist_gen_2d = hist_gen_3d.Project3D('zx')
        hist_gen_2d.SetName("hist_gen_2d")

        # Get the 0th order correction factor
        self.efficiency = r.TH1D("efficiency", "Efficiency", len(self.ptBinsArray) - 1, np.asarray(self.ptBinsArray, 'd'))
        for i, bin in enumerate(self.ptBins):
            lowerBin = hist_gen_2d.GetXaxis().FindBin(bin.lowerPt)
            upperBin = hist_gen_2d.GetXaxis().FindBin(bin.upperPt) - 1
            hist_rec_ir_orig = hist_rec_2d.ProjectionY(f"hist_rec_{bin.lowerPt}_{bin.upperPt}", firstxbin=lowerBin, lastxbin=upperBin)
            hist_gen_ir_orig = hist_gen_2d.ProjectionY(f"hist_gen_{bin.lowerPt}_{bin.upperPt}", firstxbin=lowerBin, lastxbin=upperBin)
            hist_gen_ir, hist_rec_ir, n_trimmed = utils.trim_trailing_zeros_root(hist_gen_ir_orig, hist_rec_ir_orig)
            eff_i, eff_err_i, _, _ = apply_ir_rescaling(hist_rec_ir, hist_gen_ir, theta, cov_theta, 0, n_trimmed, n_data)
            self.efficiency.SetBinContent(i+1, eff_i)
            self.efficiency.SetBinError(i+1, eff_err_i)
            self.efficiency.SetBinError(i+1, eff_err_i)

        # Create a finely binned efficiency histogram to be used in reweighting
        self.efficiencyFine = self.histD0PtMatched.Clone()
        self.efficiencyFine.Reset()
        for i in range(self.efficiencyFine.GetNbinsX()):
            hist_rec_ir_orig = hist_rec_2d.ProjectionY(f"hist_rec_{i+1}", firstxbin=i+1, lastxbin=i+1)
            hist_gen_ir_orig = hist_gen_2d.ProjectionY(f"hist_gen_{i+1}", firstxbin=i+1, lastxbin=i+1)
            hist_gen_ir, hist_rec_ir, n_trimmed = utils.trim_trailing_zeros_root(hist_gen_ir_orig, hist_rec_ir_orig)
            eff_i, eff_err_i, _, _ = apply_ir_rescaling(hist_rec_ir, hist_gen_ir, theta, cov_theta, 0, n_trimmed, n_data)
            self.efficiencyFine.SetBinContent(i+1, eff_i)
            self.efficiencyFine.SetBinError(i+1, eff_err_i)
            self.efficiencyFine.SetBinError(i+1, eff_err_i)

        # Now that the efficiency has been corrected, re-calcualte the corrected spectrum
        self.calculate_corrected_spectrum()
        
        # Plot
        if hasattr(self, 'canvasIrCorrected'):
            del self.canvasIrCorrected
        self.canvasIrCorrected = r.TCanvas("canvasIrCorrected", "canvasIrCorrected", 1400, 500)
        self.canvasIrCorrected.Divide(2, 1)
        self.canvasIrCorrected.cd(1)
        self.efficiency.Draw()
        self.efficiency.SetTitle(f"{self.efficiency.GetTitle()} (IR dep. corrected)")
        self.efficiency.GetYaxis().SetTitle("Efficiency")
        self.canvasIrCorrected.cd(2)
        self.histCorrectedSpectrum.SetStats(0)
        self.histCorrectedSpectrum.Draw()
        r.gPad.SetLogy()
        self.canvasIrCorrected.Draw()
    
    def calculate_cross_section(self, statErrorsOnly=False, draw=True):
        # Get the branching fraction from the PDG
        import pdg
        pdgApi = pdg.connect()
        pdgD0 = pdgApi.get_particle_by_name('D0')
        pdgD0KpiDecay = pdgApi.get(f'{pdgD0.baseid}.1/2026')
        branchingFractionD0Kpi = pdgD0KpiDecay.value
        print(f'Using branching fraction {branchingFractionD0Kpi} for {pdgD0KpiDecay.description}')
        # Calculate cross section
        self.histCrossSection = self.histCorrectedSpectrum.Clone()
        self.histCrossSection.SetName("histCrossSection")
        self.histCrossSection.SetTitle("D^{0} differential cross section")
        self.histCrossSection.Scale(1. / (self.maxY - self.minY)) # Divide by dy
        self.histCrossSection.Scale(1. / 2.) # Account for cc
        self.histCrossSection.Scale(1. / branchingFractionD0Kpi)
        self.histCrossSection.Scale(1. / self.lumi) # µb / GeVc^-1
        self.histCrossSection.Scale(1. / 1000.) # mb / GeVc^-1
        self.histCrossSection.GetYaxis().SetTitle("d^{2}#sigma/dp_{T}dy (mb/GeVc^{-1})")
        for i, bin in enumerate(self.ptBins):
            err = (bin.nominalFitResult.relativeStatError * self.histCrossSection.GetBinContent(i+1))**2
            if not statErrorsOnly:
                err += (bin.relativeSysError * self.histCrossSection.GetBinContent(i+1))**2
            err = np.sqrt(err)
            self.histCrossSection.SetBinError(i+1, err)
        # Create a TGraphAsymmErrors so that the asymmetric Minos errors can be shown
        self.graphCrossSection = r.TGraphAsymmErrors(self.histCrossSection)
        for i, bin in enumerate(self.ptBins):
            errLower = (bin.nominalFitResult.relativeStatErrorLower * self.graphCrossSection.GetPointY(i))**2
            errUpper = (bin.nominalFitResult.relativeStatErrorUpper * self.graphCrossSection.GetPointY(i))**2
            if not statErrorsOnly:
                errLower += (bin.relativeSysError * self.graphCrossSection.GetPointY(i))**2
                errUpper += (bin.relativeSysError * self.graphCrossSection.GetPointY(i))**2
            errLower = np.sqrt(errLower)
            errUpper = np.sqrt(errUpper)
            self.graphCrossSection.SetPointEYlow(i, errLower)
            self.graphCrossSection.SetPointEYhigh(i, errUpper)
        if draw:
            if hasattr(self, 'canvasCrossSection'):
                del self.canvasCrossSection
            self.canvasCrossSection = r.TCanvas("canvasCrossSection")
            self.canvasCrossSection.cd()
            self.graphCrossSection.Draw('ap')
            if statErrorsOnly:
                self.graphCrossSection.SetTitle(f"{self.graphCrossSection.GetTitle()} (stat. errors only)")
            self.graphCrossSection.GetYaxis().SetTitle("d^{2}#sigma/dp_{T}dy (mb/GeVc^{-1})")
            self.graphCrossSection.GetXaxis().SetTitle("p_{T} (GeV/c)")
            r.gPad.SetLogy()
            self.canvasCrossSection.Draw()

    def calculate_systematic_uncertainty(self):
        # Add in quadrature the sytematic uncertainty from all sources
        for ptBin in self.ptBins:
            relErr = 0
            # Yield extraction
            relErr += (ptBin.histSystematicYieldDistribution.GetRMS() / ptBin.histSystematicYieldDistribution.GetMean())**2
            # Other sources here...
            relErr = np.sqrt(relErr)
            ptBin.relativeSysError = relErr

    def draw_efficiency(self):
        if hasattr(self, 'canvasEfficiency'):
            del self.canvasEfficiency
        self.canvasEfficiency = r.TCanvas("canvasEfficiency", "canvasEfficiency", 1200, 666)
        self.canvasEfficiency.Divide(3, 2, 0.002, 0.01)
        self.canvasEfficiency.cd(1)
        self.histD0PtGenerated.Draw()
        self.canvasEfficiency.cd(2)
        self.histD0PtMatched.Draw()
        self.canvasEfficiency.cd(3)
        self.efficiencyFine.Draw()
        self.canvasEfficiency.cd(4)
        self.efficiency.SetStats(0)
        self.efficiency.Draw()
        if self.reweighting:
            self.efficiencyWithoutReweighting.SetStats(0)
            self.efficiencyWithoutReweighting.Draw("same")
            self.legendEfficiency4 = r.TLegend(0.55, 0.15, 0.9, 0.3)
            self.legendEfficiency4.AddEntry(self.efficiency, "With reweighting")
            self.legendEfficiency4.AddEntry(self.efficiencyWithoutReweighting, "Without reweighting")
            self.legendEfficiency4.SetBorderSize(0)
            self.legendEfficiency4.SetFillStyle(0)
            self.legendEfficiency4.Draw()
        self.canvasEfficiency.cd(5)
        r.gPad.SetLogy()
        self.histCorrectedSpectrum.SetStats(0)
        self.histCorrectedSpectrum.Draw()
        if self.reweighting:
            self.histCorrectedSpectrumWithoutReweighting.SetStats(0)
            self.histCorrectedSpectrumWithoutReweighting.Draw("same")
            self.reweightingFunc.Draw("same")
            self.legendEfficiency5 = r.TLegend(0.55, 0.7, 0.9, 0.85)
            self.legendEfficiency5.AddEntry(self.histCorrectedSpectrum, "With reweighting")
            self.legendEfficiency5.AddEntry(self.histCorrectedSpectrumWithoutReweighting, "Without reweighting")
            self.legendEfficiency5.SetBorderSize(0)
            self.legendEfficiency5.SetFillStyle(0)
            self.legendEfficiency5.Draw()
        # Create histogram showing ratio before / after reweighting
        if self.reweighting:
            self.histCorrectedSpectrumRatioAfterBeforeReweighting = self.histCorrectedSpectrum.Clone()
            self.histCorrectedSpectrumRatioAfterBeforeReweighting.Divide(self.histCorrectedSpectrumWithoutReweighting)
            self.histCorrectedSpectrumRatioAfterBeforeReweighting.SetName("histCorrectedSpectrumRatioAfterBeforeReweighting")
            self.histCorrectedSpectrumRatioAfterBeforeReweighting.SetTitle("Corrected spectrum after reweighting / before reweighting")
            self.histCorrectedSpectrumRatioAfterBeforeReweighting.GetYaxis().SetTitle("Ratio")
            self.histCorrectedSpectrumRatioAfterBeforeReweighting.SetStats(0)
            self.canvasEfficiency.cd(6)
            self.histCorrectedSpectrumRatioAfterBeforeReweighting.Draw()

        self.canvasEfficiency.Draw()


    def draw_fits_and_yield(self, showFitRangeOnly=False, nCols=3, showRawYield=True, showLegend=False, componentStyle=1, prependTitle="", width=400, height=333):
        # Figure out grid layout
        nPanels = len(self.ptBins) + showRawYield
        nRows = int(np.ceil(nPanels/nCols))
        if hasattr(self, 'canvasFitsYields'):
            del self.canvasFitsYields
        self.canvasFitsYields = r.TCanvas("canvasFitsYields", "canvasFitsYields", nCols * width, nRows * height)
        self.canvasFitsYields.Divide(nCols, nRows, 0.002, 0.01)

        self.textBoxesFitsYields = []
        self.linesFitrangeLow = []
        self.linesFitrangeHigh = []
        self.legendsFitsYields = []
        for i, bin in enumerate(self.ptBins):
            self.histRawYield.SetBinContent(i+1, bin.nominalFitResult.nSignal / self.histRawYield.GetBinWidth(i+1))
            # Error propagation with the bin width
            self.histRawYield.SetBinError(i+1, bin.nominalFitResult.relativeStatError / self.histRawYield.GetBinWidth(i+1) * bin.nominalFitResult.nSignal)
            # Draw the histograms with fits
            pad = self.canvasFitsYields.cd(i+1)
            pad.SetLeftMargin(0.125)
            pad.SetRightMargin(0.025)
            pad.SetBottomMargin(0.11)
            pad.SetTopMargin(0.09)
            self.legendsFitsYields.append(r.TLegend(0.15, 0.15, 0.43 if bin.nominalFitResult.corrBkgFunc is None else 0.53, 0.43))
            self.legendsFitsYields[i].SetBorderSize(0)
            self.legendsFitsYields[i].SetTextSize(0.04)
            self.legendsFitsYields[i].SetFillStyle(0)
            self.legendsFitsYields[i].SetFillColor(0)
            self.legendsFitsYields[i].SetLineColor(0)
            r.TGaxis.SetMaxDigits(3)
            bin.massPtSlice.Draw("E")
            bin.massPtSlice.SetTitle(prependTitle + f"{bin.lowerPt} #leq p_{{ T}} < {bin.upperPt} GeV/c")
            bin.massPtSlice.GetXaxis().SetTitle("m_{K#pi} (GeV/c^{2})")
            bin.massPtSlice.GetYaxis().SetTitle(f"Counts per {bin.massPtSlice.GetBinWidth(1)*1000:.0f} MeV/c^{{2}}")
            bin.massPtSlice.GetXaxis().SetTitleSize(0.05)
            bin.massPtSlice.GetYaxis().SetTitleSize(0.05)
            bin.massPtSlice.GetXaxis().SetLabelSize(0.04)
            bin.massPtSlice.GetYaxis().SetLabelSize(0.04)
            bin.massPtSlice.SetLineColor(r.kBlack)
            bin.massPtSlice.SetMarkerStyle(r.kFullCircle)
            bin.massPtSlice.SetMarkerSize(0.5)
            bin.massPtSlice.SetStats(0)
            bin.massPtSlice.SetMinimum(0.0)
            if showFitRangeOnly:
                bin.massPtSlice.GetXaxis().SetRangeUser(bin.nominalFitResult.lowerMass, bin.nominalFitResult.upperMass)
            else:
                bin.massPtSlice.GetXaxis().SetRangeUser(1.5, 2.2)
            self.legendsFitsYields[i].AddEntry(bin.massPtSlice, "Data")
            bin.nominalFitResult.fitFunc.Draw("same")
            self.legendsFitsYields[i].AddEntry(bin.nominalFitResult.fitFunc, "Total fit function", "l")
            bin.nominalFitResult.backgroundFunc.Draw("same")
            self.legendsFitsYields[i].AddEntry(bin.nominalFitResult.backgroundFunc, f"Comb. background", "l")
            if componentStyle == 1:
                # Draw the sum of the combinatorial background and reflected background on an existing canvas
                def make_comb_plus_refl_callable(bbin):
                    def f(x: np.ndarray, par: np.ndarray) -> float:
                        m = x[0]
                        return bbin.nominalFitResult.backgroundFunc.Eval(m) + bbin.nominalFitResult.dataReflFunc.Eval(m)
                    return f
                fCall = make_comb_plus_refl_callable(bin)
                _tf1_callables.append(fCall)
                funcCombPlusRefl = r.TF1(f"combPlusReflFunc_bin{bin.index}", _tf1_callables[-1], bin.nominalFitResult.backgroundFunc.GetXmin(), bin.nominalFitResult.backgroundFunc.GetXmax(), 0)
                funcCombPlusRefl.SetLineColor(r.kGreen+1)
                funcCombPlusRefl.SetLineStyle(r.kDashed)
                funcCombPlusRefl.SetNpx(1000)
                funcCombPlusRefl.Draw('same')
                self.legendsFitsYields[i].AddEntry(funcCombPlusRefl, f"Comb. + refl bkg", "l")
                if bin.nominalFitResult.corrBkgFunc is not None:
                    # bin.nominalFitResult.draw_combpluscorrbkgfunc()
                    def make_comb_plus_corr_callable(bbin):
                        def g(x: np.ndarray, par: np.ndarray) -> float:
                            m = x[0]
                            return bbin.nominalFitResult.backgroundFunc.Eval(m) + bbin.nominalFitResult.corrBkgFunc.Eval(m)
                        return g
                    gCall = make_comb_plus_corr_callable(bin)
                    _tf1_callables.append(gCall)
                    funcCombPlusCorr = r.TF1(f"combPlusCorrBkgFunc_bin{bin.index}", _tf1_callables[-1], bin.nominalFitResult.backgroundFunc.GetXmin(), bin.nominalFitResult.backgroundFunc.GetXmax(), 0)
                    funcCombPlusCorr.SetLineColor(r.kOrange+1)
                    funcCombPlusCorr.SetLineStyle(r.kDashed)
                    funcCombPlusCorr.SetNpx(1000)
                    funcCombPlusCorr.Draw('same')
                    self.legendsFitsYields[i].AddEntry(funcCombPlusCorr, "Comb. bkg + D^{0}#rightarrow K^{-}#pi^{+}#pi^{0}", "l")
            elif componentStyle == 2:
                bin.nominalFitResult.signalFunc.Draw("same LF2")
                self.legendsFitsYields[i].AddEntry(bin.nominalFitResult.signalFunc, "Signal", "l")
                bin.nominalFitResult.dataReflFunc.Draw("same")
                self.legendsFitsYields[i].AddEntry(bin.nominalFitResult.dataReflFunc, f"Refl. background", "l")
                if bin.nominalFitResult.corrBkgFunc is not None:
                    bin.nominalFitResult.corrBkgFunc.Draw("same")
                    self.legendsFitsYields[i].AddEntry(bin.nominalFitResult.corrBkgFunc, "D^{0}#rightarrow K^{-}#pi^{+}#pi^{0}", "l")
            # Draw data again so that it sits in front
            bin.massPtSlice.Draw("E same")
            if showLegend:
                self.legendsFitsYields[i].Draw()
            # Text box with info
            self.textBoxesFitsYields.append(r.TPaveText(0.71, 0.55, 0.95, 0.9, "NDC"))
            self.textBoxesFitsYields[i].SetName(f"textbox_bin{i}")
            self.textBoxesFitsYields[i].SetFillColor(0)
            self.textBoxesFitsYields[i].SetFillStyle(0)
            self.textBoxesFitsYields[i].SetBorderSize(0)
            self.textBoxesFitsYields[i].SetTextAlign(12)
            self.textBoxesFitsYields[i].SetTextFont(42)
            self.textBoxesFitsYields[i].SetTextSize(0.04)
            self.textBoxesFitsYields[i].AddText(f"#mu = {bin.nominalFitResult.fitMu:.3f}")
            self.textBoxesFitsYields[i].AddText(f"#sigma = {bin.nominalFitResult.fitSigma:.4f}")
            self.textBoxesFitsYields[i].AddText(f"S = {bin.nominalFitResult.nSignal:.0f}^{{+{bin.nominalFitResult.relativeStatErrorUpper*bin.nominalFitResult.nSignal:.0f}}}_{{-{bin.nominalFitResult.relativeStatErrorLower*bin.nominalFitResult.nSignal:.0f}}}")
            self.textBoxesFitsYields[i].AddText(f"S/B (3#sigma) = {bin.nominalFitResult.signalToBackground:.3f}")
            self.textBoxesFitsYields[i].AddText(f"S/#sqrt{{S+B}} = {bin.nominalFitResult.signalSignificance:.1f}")
            # self.textBoxesFitsYields[i].AddText(f"Refl/S = {bin.nominalFitResult.nReflBackground/bin.nominalFitResult.nSignal:.3f}")
            self.textBoxesFitsYields[i].AddText(f"#chi^{{2}}/ndf = {bin.nominalFitResult.fitChi2Ndf:.3f}")
            self.textBoxesFitsYields[i].Draw()

        # Draw raw yield histogram
        if showRawYield:
            self.canvasFitsYields.cd(nPanels)
            self.histRawYield.Draw()

        self.canvasFitsYields.Draw()

        if not showFitRangeOnly:
            for i, bin in enumerate(self.ptBins):
                self.canvasFitsYields.cd(i+1)
                # Lines to show fitting range
                self.linesFitrangeLow.append(r.TLine(bin.nominalFitResult.lowerMass, r.gPad.GetUymin(), bin.nominalFitResult.lowerMass, r.gPad.GetUymax()))
                self.linesFitrangeLow[i].SetLineColor(r.kBlack)
                self.linesFitrangeLow[i].SetLineStyle(2)
                self.linesFitrangeLow[i].SetLineWidth(1)
                self.linesFitrangeLow[i].Draw()
                self.linesFitrangeHigh.append(r.TLine(bin.nominalFitResult.upperMass, r.gPad.GetUymin(), bin.nominalFitResult.upperMass, r.gPad.GetUymax()))
                self.linesFitrangeHigh[i].SetLineColor(r.kBlack)
                self.linesFitrangeHigh[i].SetLineStyle(2)
                self.linesFitrangeHigh[i].SetLineWidth(1)
                self.linesFitrangeHigh[i].Draw()

    def draw_reflected_fits(self, nCols=3, width=400, height=333):
        # Figure out grid layout
        nPanels = len(self.ptBins)
        nRows = int(np.ceil(nPanels/nCols))
        if hasattr(self, 'canvasReflFits'):
            del self.canvasReflFits
        self.canvasReflFits = r.TCanvas("canvasReflFits", "canvasReflFits", nCols * width, nRows * height)
        self.canvasReflFits.Divide(nCols, nRows, 0.002, 0.01)

        self.textBoxesReflected = []
        for i, bin in enumerate(self.ptBins):
            pad = self.canvasReflFits.cd(i+1)
            bin.massPtSliceReflected.Draw("E")
            bin.massPtSliceReflected.SetStats(0)
            bin.massPtSliceReflected.SetTitle(f"{self.ptBins[i].lowerPt} #leq p_{{T}} < {self.ptBins[i].upperPt} GeV/c")
            bin.massPtSliceReflected.SetTitleSize(0.05)
            bin.massPtSliceReflected.GetXaxis().SetTitleSize(0.05)
            bin.massPtSliceReflected.GetYaxis().SetTitleSize(0.05)
            bin.massPtSliceReflected.GetXaxis().SetLabelSize(0.04)
            bin.massPtSliceReflected.GetYaxis().SetLabelSize(0.04)
            bin.massPtSliceReflected.GetYaxis().SetTitle(f"Counts per {bin.massPtSliceReflected.GetBinWidth(1)*1000:.0f} MeV/c^{{2}}")
            bin.massPtSliceReflected.GetXaxis().SetTitle("Mass (GeV/c^{2})")
            pad.SetLeftMargin(0.11)
            pad.SetRightMargin(0.02)
            pad.SetBottomMargin(0.12)
            pad.SetTopMargin(0.08)
            bin.reflFunc1 = r.TF1(f"fReflGauss1_bin{bin.index}", fit_functions.signal, 1.3, 2.3, 3)
            bin.reflFunc1.SetParameters(np.array([bin.reflFunc.GetParameter(0) * bin.reflFunc.GetParameter(1),
                                                  bin.reflFunc.GetParameter(2), bin.reflFunc.GetParameter(3)], dtype='d'))
            bin.reflFunc1.SetLineColor(r.kGreen)
            bin.reflFunc1.SetLineStyle(r.kDashed)
            bin.reflFunc1.Draw("same")
            bin.reflFunc2 = r.TF1(f"fReflGauss2_bin{bin.index}", fit_functions.signal, 1.3, 2.3, 3)
            bin.reflFunc2.SetParameters(np.array([bin.reflFunc.GetParameter(0) * (1 - bin.reflFunc.GetParameter(1)),
                                                  bin.reflFunc.GetParameter(4), bin.reflFunc.GetParameter(5)], dtype='d'))
            bin.reflFunc2.SetLineColor(r.kMagenta)
            bin.reflFunc2.SetLineStyle(r.kDashed)
            bin.reflFunc2.Draw("same")
            # bin.reflFuncSum = r.TF1(f"fReflGauss1Gauss2_bin{bin}", fit_functions.reflected_background, 1.3, 2.3, 6)
            # bin.reflFuncSum.SetParameters(np.asarray(bin.reflFunc.GetParameters()))
            # bin.reflFuncSum.SetLineColor(r.kRed)
            bin.reflFunc.SetLineColor(r.kRed)
            bin.reflFunc.Draw("same")
            # Text box with info
            self.textBoxesReflected.append(r.TPaveText(0.71, 0.4, 0.96, 0.9, "NDC"))
            self.textBoxesReflected[i].SetName(f"textboxreflected_bin{i}")
            self.textBoxesReflected[i].SetFillColor(0)
            self.textBoxesReflected[i].SetFillStyle(0)
            self.textBoxesReflected[i].SetBorderSize(0)
            self.textBoxesReflected[i].SetTextAlign(12)
            self.textBoxesReflected[i].SetTextFont(42)
            self.textBoxesReflected[i].SetTextSize(0.035)
            self.textBoxesReflected[i].AddText(f"Refl/S = {bin.reflectedRatio:.3f}")
            # self.textBoxesReflected[i].AddText(f"chi2/ndof = {bin.reflChi2Ndf:.3f}")
            self.textBoxesReflected[i].AddText(f"Integral = {bin.reflFunc.GetParameter('Integral'):.3f}")
            self.textBoxesReflected[i].AddText(f"RelNorm = {bin.reflFunc.GetParameter('RelNorm'):.3f}")
            self.textBoxesReflected[i].AddText(f"Mean1 = {bin.reflFunc.GetParameter('Mean1'):.3f}")
            self.textBoxesReflected[i].AddText(f"Sigma1 = {bin.reflFunc.GetParameter('Sigma1'):.3f}")
            self.textBoxesReflected[i].AddText(f"Mean2 = {bin.reflFunc.GetParameter('Mean2'):.3f}")
            self.textBoxesReflected[i].AddText(f"Sigma2 = {bin.reflFunc.GetParameter('Sigma2'):.3f}")
            self.textBoxesReflected[i].Draw()

        self.canvasReflFits.Draw()

    def closure_test(self, truth_input, dir_response=None):
        """
        Perform a closure test of the efficiency correction procedure
        truth_input  : numpy array of D0 pT spectrum truth values
        dir_response : Directory of AnalysisResults files containing histograms used to construct
                      the forward model
        """
        if not (hasattr(self, 'histTruthPt') and hasattr(self, 'histResponse')):
            hist_name_gen = f"analysis-asymmetric-pairing/output;1/{self.groupNameD0Generated}/MyMcPtHisto"
            hist_name_response = f"analysis-asymmetric-pairing/output;1/{self.groupNameD0PtMatched}/MyMCPtPtHisto"
            hist_dict = get_histograms(dir_response, self.runList, [hist_name_gen, hist_name_response]) 
            self.histResponse = hist_dict[hist_name_response]
            self.histTruthPt = hist_dict[hist_name_gen]
        if not (hasattr(self, 'responseMatrix')):
            self.calculate_response_matrix(dir_response)
        edges = self.ptBinsArray.copy()
        edges.append(20.0)
        edges = np.array(edges, dtype='d')
        h_truth = self.histTruthPt.Rebin(len(edges)-1, "h_truth", edges)

        # Create efficiency per truth bin to use in forward model
        # Number of MC events within each truth bin which were reconstructed and selected somewhere
        N_MC_reco = np.zeros(len(edges)-1)
        h_response_proj_reco = h_response.ProjectionX()
        for i in range(len(edges)-1):
            N_MC_reco[i] = h_response_proj_reco.GetBinContent(i+1)
        # Number of MC events within each truth bin
        N_MC_truth = np.zeros(len(edges)-1)
        for i in range(len(edges)-1):
            N_MC_truth[i] = h_truth.GetBinContent(i+1)
        # Absolute efficiency per truth bin
        eff_truth = N_MC_reco / N_MC_truth
        print("Truth efficiency (N rec in truth bin i / N gen in truth bin i)")
        print(eff_truth)

        # Pass truth input through forward model to create pseudo-data
        pseudo_data = response_matrix @ (truth_input * eff_truth)

        # Get the pseudodata and truth input into histogram form and convert to dN/dpT
        hist_pseudo = r.TH1D("hist_pseudo", "hist_pseudo", len(edges)-2, edges[:-1])
        hist_pseudo.GetXaxis().SetTitle("pT (GeV/c)")
        hist_pseudo.GetYaxis().SetTitle("dN/dpT (1/GeV/c^{-1})")
        hist_truth_input = r.TH1D("hist_truth_input", "hist_truth_input", len(edges)-2, edges[:-1])
        hist_truth_input.GetXaxis().SetTitle("pT (GeV/c)")
        hist_truth_input.GetYaxis().SetTitle("dN/dpT (1/GeV/c^{-1})")
        for i in range(hist_pseudo.GetNbinsX()):
            hist_pseudo.SetBinContent(i+1, pseudo_data[i] / hist_pseudo.GetBinWidth(i+1))
            hist_truth_input.SetBinContent(i+1, truth_input[i] / hist_truth_input.GetBinWidth(i+1))

        # Recreate the steps in the correction procedure
        # Apply initial bin-by-bin efficiency
        hist_pseudo_corrected0 = hist_pseudo.Clone()
        hist_pseudo_corrected0.SetName("hist_pseudo_corrected0")
        hist_pseudo_corrected0.Divide(self.efficiencyWithoutReweighting)
        # Apply reweighting procedure
        reweightingFunc = r.TF1("powerLaw", "[0]*x/TMath::Power((1+TMath::Power(x/[1],[3])),[2])", 0, 12)
        reweightingFunc.SetParameters(394000, 1.83, 1.78, 2.87)
        print("===== Fitting to the corrected spectrum for reweighting =====")
        reweightingFitResults = hist_pseudo_corrected0.Fit(reweightingFunc, "LS")
        efficiencyReweighted = self.efficiencyWithoutReweighting.Clone()
        efficiencyReweighted.Reset()
        efficiencyReweighted.SetName("efficiencyReweighted")
        efficiencyReweighted.SetTitle("(Reconstructed / Generated) (reweighted)")
        dpT = self.efficiencyFine.GetBinWidth(1)
        # Calculate numerator of <epsilon>_i
        for i in range(1, self.efficiencyFine.GetNbinsX() + 1):
            binCenter = self.efficiencyFine.GetBinCenter(i)
            binContent = self.efficiencyFine.GetBinContent(i)
            binError = self.efficiencyFine.GetBinError(i)
            weight = reweightingFunc.Eval(binCenter)
            # Calculate weighted content
            weightedContent = binContent * weight * dpT
            weightedError = np.abs(weight) * dpT * binError # Error propagation
            # Find the target bin and fill it
            targetBin = efficiencyReweighted.FindBin(binCenter)
            efficiencyReweighted.AddBinContent(targetBin, weightedContent)
            currentErrorTarget = efficiencyReweighted.GetBinError(targetBin)
            efficiencyReweighted.SetBinError(targetBin, np.sqrt(currentErrorTarget**2 + weightedError**2))
        # Calculate denominator of <epsilon>_i
        histDenominator = self.efficiencyWithoutReweighting.Clone()
        histDenominator.Reset()
        # For some reason the first iteration of IntegralError gives a nonsensical value, throw this away first
        _ = reweightingFunc.IntegralError(0., 0.1, reweightingFitResults.GetParams(), reweightingFitResults.GetCovarianceMatrix().GetMatrixArray(), epsilon=1e-8)
        for i in range(1, efficiencyReweighted.GetNbinsX() + 1):
            lowEdge = efficiencyReweighted.GetXaxis().GetBinLowEdge(i)
            upEdge = efficiencyReweighted.GetXaxis().GetBinUpEdge(i)
            print(f"--- Calculating integral of reweightingFunc from {lowEdge} to {upEdge} ---")
            integral = reweightingFunc.Integral(lowEdge, upEdge)
            histDenominator.SetBinContent(i, integral)
            integralError = reweightingFunc.IntegralError(lowEdge, upEdge, reweightingFitResults.GetParams(), reweightingFitResults.GetCovarianceMatrix().GetMatrixArray(), epsilon=1e-8)
            print(f"Bin {i}: integral = {integral} +- {integralError}")
            histDenominator.SetBinError(i, integralError)
        # Obtain <epsilon>_i as a histogram
        efficiencyReweighted.Divide(histDenominator)
        # Apply reweighted correction factor
        hist_pseudo_corrected1 = hist_pseudo.Clone()
        hist_pseudo_corrected1.SetName("hist_pseudo_corrected1")
        hist_pseudo_corrected1.SetTitle("Corrected pT spectrum with reweighting")
        hist_pseudo_corrected1.GetYaxis().SetTitle("dN/dp_{T}")
        hist_pseudo_corrected1.Divide(efficiencyReweighted)

        # Corrected, reweighted pseudodata is now compared to input truth
        pseudo_corrected1 = np.zeros_like(truth_input[:-1])
        for i in range(len(pseudo_corrected1)):
            pseudo_corrected1[i] = hist_pseudo_corrected1.GetBinContent(i+1) * hist_pseudo_corrected1.GetBinWidth(i+1)
        delta = (pseudo_corrected1 - truth_input[:-1]) / truth_input[:-1]
        print(f"delta = {delta}")

        if hasattr(self, 'canvasClosureTest'):
            del self.canvasClosureTest
        self.canvasClosureTest = r.TCanvas(f"canvasClosureTest", f"canvasClosureTest")
        self.canvasClosureTest.cd()
        _tf1_callables.append(hist_pseudo_corrected0)
        hist_pseudo_corrected0.Draw()
        hist_pseudo_corrected0.SetStats(0)
        hist_pseudo_corrected0.SetLineColor(r.kRed)
        _tf1_callables.append(hist_pseudo_corrected1)
        hist_pseudo_corrected1.Draw("same")
        hist_pseudo_corrected1.SetLineColor(r.kGreen-1)
        _tf1_callables.append(hist_pseudo)
        hist_pseudo.Draw("same")
        hist_pseudo.SetLineColor(r.kBlue)
        _tf1_callables.append(hist_truth_input)
        hist_truth_input.Draw("same")
        hist_truth_input.SetLineColor(r.kMagenta)
        self.canvasClosureTest.Draw()

    def calculate_response_matrix(self, dir_response=None):
        if not (hasattr(self, 'histResponse')):
            if dir_response is None:
                dir_response = self.dirRec
            hist_name_response = f"analysis-asymmetric-pairing/output;1/{self.groupNameD0PtMatched}/MyMCPtPtHisto"
            hist_dict = get_histograms(dir_response, self.runList, [hist_name_response]) 
            self.histResponse = hist_dict[hist_name_response]
        edges = self.ptBinsArray.copy()
        edges.append(20.0)
        edges = np.array(edges, dtype='d')
        h_response = r.TH2D("h_response", "h_response", len(edges)-1, edges, len(edges)-1, edges)
        h_response = utils.rebin_th2_to_reference(self.histResponse, h_response, "")

        # Construct the response matrix
        response_matrix = np.zeros((len(edges)-1, len(edges)-1))
        for i in range(len(edges)-1):
            # Column i
            column_sum = 0
            for j in range(len(edges)-1):
                # Row j
                content = h_response.GetBinContent(i+1, j+1)
                response_matrix[j, i] = content
                column_sum += content
            # Normalize column-wise
            response_matrix[:, i] /= column_sum
        print("Response matrix:")
        with np.printoptions(precision=4, suppress=True, linewidth=100, threshold=1000):
            print(response_matrix)
        self.responseMatrix = response_matrix
        self.histResponseFinalBins = h_response

    def draw_response_matrix(self, savefig=None):
        plt.rcParams.update({
            "text.usetex": True,
        })
        fig, ax = plt.subplots(figsize=(6.5,6.5))
        white_green_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("WhiteGreen", ["white", "yellow", "lightgreen"])
        im = ax.matshow(self.responseMatrix, cmap=white_green_cmap, norm=matplotlib.colors.SymLogNorm(0.08))
        # Get colormap and normalizer from the image
        cmap = im.get_cmap()
        norm = im.norm
        # Add text inside cells; choose white/black based on background brightness
        for (i, j), z in np.ndenumerate(self.responseMatrix):
            # Get RGBA color for this cell
            rgba = cmap(norm(z))
            r, g, b, _ = rgba
            # Compute perceived luminance
            # (standard formula: 0.299 R + 0.587 G + 0.114 B)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            # If background is dark, use white text; otherwise black
            text_color = 'white' if luminance < 0.5 else 'black'
            ax.text(
                j, i,
                '{:0.3f}'.format(z) if z!=0 else '{:.0f}'.format(z),
                ha='center', va='center',
                color=text_color
            )
            
        # Number of rows and columns
        n_rows, n_cols = self.responseMatrix.shape
        # Set ticks at integer positions
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        # Make tick labels start at 1 instead of 0
        ax.set_xticklabels(np.arange(1, n_cols + 1))
        ax.set_yticklabels(np.arange(1, n_rows + 1))
        # Axis titles
        ax.set_xlabel('Truth $p_\mathrm{T}$ bin', fontsize=15, labelpad=10)
        ax.xaxis.set_label_position('top')   # move label to top
        ax.tick_params(top=True, bottom=False)  # optional: show only top ticks
        ax.set_ylabel('Reconstructed $p_\mathrm{T}$ bin', fontsize=15, labelpad=10)

        if savefig is not None:
            plt.savefig(savefig)
        plt.show()

class McAnalysis():
    """Class for analysing the MC only"""

    def __init__(self, dirRec, dirGen = None,
                 ptBins = [0., 0.75, 1.5, 2.25, 3., 4., 6., 8., 12.], runList = None, 
                 recFileStructure=FileStructures.FLAT,
                 genFileStructure=FileStructures.FLAT,
                 **kwargs):
        # The runList defines which files are looped over in run-by-run calculations
        self.runList = runList if runList is not None else DEFAULT_RUNLIST_PASS4.copy()
        self.runList = sorted(self.runList)
        print(f"This MC analysis contains {len(self.runList)} runs")

        self.dirRec = dirRec
        self.dirGen = dirGen

        # Get default cut names, replace if specified in construction
        cutNames = {
            "kaonLegCutName" : "kaonPIDTPCTOFpTDCAz",
            "pionLegCutName" : "pionNoPIDpTDCAz",
            "pairCutName"    : "PtDepLxyCosPointingAngleCut"
        }
        cutNames.update(kwargs)
        self.kaonLegCutName = cutNames["kaonLegCutName"]
        self.pionLegCutName = cutNames["pionLegCutName"]
        self.pairCutName = cutNames["pairCutName"]
        if self.pairCutName != "":
            self.pairCutName = "_" + self.pairCutName
        # pT bins to be used in the differential cross section
        self.ptBinsArray = ptBins
        self.ptBins = []
        for i in range(len(self.ptBinsArray) - 1):
            self.ptBins.append(PtBin(i, self.ptBinsArray[i], self.ptBinsArray[i+1]))
        # Range of rapidity bin to be used
        self.minY = -0.9
        self.maxY = 0.9
        # Get the mass histograms needed for the signal extraction
        self.recFileStructure = recFileStructure
        self.genFileStructure = genFileStructure
        self.prepare_histograms()

    def prepare_histograms(self):
        self.groupNameD0Generated = "MCTruthGenAfterBcCuts_D0FS"
        self.groupNameD0PtMatched = f"PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4{self.pairCutName}_KPiFromD0FS"

        # Main gen. lvl. histogram used for efficiency, and also all the histograms used for factorized efficiencies later
        fullNamesGen = ["MCTruthGenRec_D0FS", "MCTruthGenSel_D0FS", "MCTruthGenSelDaughtersInAcc_KPiFromD0FS", "MCTruthGenRecDaughtersInAcc_KPiFromD0FS", "MCTruthGenAfterBcCutsDaughtersInAcc_KPiFromD0FS"]
        fullNamesGen = ["analysis-asymmetric-pairing/output;1/" + name + "/PtMC_YMC" for name in fullNamesGen]
        fullNameHistD0PtYGenerated = "analysis-asymmetric-pairing/output;1/" + self.groupNameD0Generated + "/MyMcPtYHisto"
        fullNamesGen.append(fullNameHistD0PtYGenerated)
        # Main rec. matched histogram, and the rec. lvl. histograms used for factorized efficiencies later
        fullNamesMc = ["noTrackCut:noTrackCut", f"{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4", f"{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4{self.pairCutName}"]
        fullNamesMc = ["analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_" + fullNameRec + "_KPiFromD0FS/Y_PtFine" for fullNameRec in fullNamesMc]
        groupNameD0Matched = "analysis-asymmetric-pairing/output;1/" + self.groupNameD0PtMatched
        fullNameHistD0YPtMatched = "analysis-asymmetric-pairing/output;1/" + self.groupNameD0PtMatched + "/Y_PtFine"
        fullNameHistD0PtMatched = "analysis-asymmetric-pairing/output;1/" + self.groupNameD0PtMatched + "/Pt"
        fullNamesMc.append(fullNameHistD0YPtMatched)

        # Get the histograms from the AnalysisResults.root files
        self.dictMcHists = {}
        if self.dirGen is not None:
            self.dictGenHists = get_histograms(self.dirGen, self.runList, fullNamesGen, histogramNamesfileStructure=self.genFileStructure)
            print("INFO: Separate directory for MC files with generator lvl. histograms was specified")
        else:
            fullNamesMc += fullNamesGen
            self.dictGenHists = self.dictMcHists

        # This dict always contains reco lvl. histograms, sometimes the gen lvl. histograms
        self.dictMcHists.update(get_histograms(self.dirRec, self.runList, fullNamesMc, histogramNamesfileStructure=self.recFileStructure))
        self.dictRecHists = self.dictMcHists

        # Fetch the histograms from the dicts
        self.histD0YPtMatched = self.dictRecHists[fullNameHistD0YPtMatched]
        self.histD0PtMatched = project_fiducial_acceptance(self.histD0YPtMatched, self.minY, self.maxY)
        self.histD0PtMatched.SetName("histD0PtMatched")
        self.histD0PtMatched.SetTitle(f"Reconstructed, matched D0 in {self.minY:.1f} < y < {self.maxY:.1f}")

        self.histD0PtYGenerated = self.dictGenHists[fullNameHistD0PtYGenerated]
        # Project out our dy bin from the gen histogram
        lowerYBin = self.histD0PtYGenerated.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenerated.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenerated = self.histD0PtYGenerated.ProjectionX(f"projPtMcGen_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenerated.SetTitle(f"Generated D0 after BC cuts in {self.minY:.1f} < y < {self.maxY:.1f}")

    
    def calculate_factorized_efficiencies(self):
        Analysis.calculate_factorized_efficiencies(self)
