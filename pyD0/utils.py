import uuid
from scipy.optimize import minimize_scalar, brentq
import ROOT as r
import math
import uproot
import seaborn as sns
import numpy as np
from random import randint
from array import array
from scipy.stats import poisson
import matplotlib.pyplot as plt
from boost_histogram import axis
import hist
from . import fit_functions

_root_callables = []

defaultITSROFlength = 14821e-9 # seconds

def lowest_safe_hf_mass(pt):
    """
    Given the HF tree in train 586890, return the lowest mass which seems safe within the pT range [minPt, maxPt],
    from an inspection of mass vs pT histogram
    """
    if pt < 3.0:
        return 1.665
    elif pt < 5.0:
        return 1.655
    elif pt < 6.0:
        return 1.465
    else:
        return 1.37


def get_mass_histogram_Kpipi0(df, minPt, maxPt):
    df = df.Filter("fFlagMc == -2 || fFlagMc == 2") # DecayChannelMain::D0ToPiKPi0
    df = df.Filter(f"fPt >= {minPt} && fPt < {maxPt}")
    h = df.Histo1D(
        (f"hCorrBkg_{minPt:.2f}_{maxPt:.2f}",
         "Kpi invariant mass, reco D0 -> Kpipi0;Counts;fM",
         150, 1.0, 2.5),
        "fM"
    )
    return h.GetValue()

def get_mass_pt_histogram_Kpipi0(df):
    df = df.Filter("fFlagMc == -2 || fFlagMc == 2") # DecayChannelMain::D0ToPiKPi0
    # h2_ptr = df.Histo2D(("h2", "title;x;y", 50, 0., 1., 40, -2., 2.), "x", "y")
    h = df.Histo2D(
        (f"hCorrBkg",
         "Kpi invariant mass, reco D0 -> Kpipi0;fM;fPt",
         1000, 1.3, 2.3,
         960, 0.0, 12.0),
        "fM", "fPt"
    )
    return h.GetValue()


def read_hf_tree(file_name):
    tree_name = "O2hfcandd0lite"

    chain = r.TChain(tree_name)

    f = r.TFile.Open(file_name)
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TDirectoryFile") and key.GetName().startswith("DF_"):
            dname = key.GetName()
            chain.Add(f"{file_name}/{dname}/{tree_name}")

    print(f"Read {chain.GetEntries()} entries from {file_name}/{tree_name}")
    return r.RDataFrame(chain)

def propagate_error_product(A, B, errA, errB, covAB):
    # Error propagation for the product of two variables r = A*B
    var = (B*errA)**2 + (A*errB)**2 + 2*A*B*covAB
    return np.sqrt(var)

def flip_th2_axes(h):
    hf = r.TH2F(f'{h.GetName()}_flip', f'{h.GetName()}_flip', h.GetNbinsY(), h.GetYaxis().GetXmin(), h.GetYaxis().GetXmax(), h.GetNbinsX(), h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax())
    hf.GetXaxis().SetTitle(h.GetYaxis().GetTitle())
    hf.GetYaxis().SetTitle(h.GetXaxis().GetTitle())
    for i in range(h.GetNbinsX() + 1):
        for j in range(h.GetNbinsY() + 1):
            hf.SetBinContent(j, i, h.GetBinContent(i, j))
            hf.SetBinError(j, i, h.GetBinError(i, j))
    return hf

def rebin_th2_to_reference(h_src, h_ref, name_suffix):
    """
    Create a clone of h_ref's binning and fill it from h_src
    by matching (x,y) bin centers.
    """
    # Create output histogram with same binning as h_ref
    name = h_ref.GetName() + name_suffix
    title = h_ref.GetTitle()

    xaxis = h_ref.GetXaxis()
    yaxis = h_ref.GetYaxis()

    # Extract bin edges from reference
    import array
    x_edges = array.array("d", [
        xaxis.GetBinLowEdge(i)
        for i in range(1, xaxis.GetNbins() + 2)
    ])
    y_edges = array.array("d", [
        yaxis.GetBinLowEdge(i)
        for i in range(1, yaxis.GetNbins() + 2)
    ])

    h_out = r.TH2D(
        name, title,
        len(x_edges) - 1, x_edges,
        len(y_edges) - 1, y_edges
    )
    h_out.Sumw2()

    # Optional: detach from any directory (safer in PyROOT)
    h_out.SetDirectory(0)

    # Loop over source bins (excluding under/overflow)
    nx_src = h_src.GetNbinsX()
    ny_src = h_src.GetNbinsY()

    for ix in range(1, nx_src + 1):
        x_center = h_src.GetXaxis().GetBinCenter(ix)
        for iy in range(1, ny_src + 1):
            y_center = h_src.GetYaxis().GetBinCenter(iy)

            content = h_src.GetBinContent(ix, iy)
            error   = h_src.GetBinError(ix, iy)

            if content == 0 and error == 0:
                continue

            # Find target bin in reference binning
            new_ix = h_out.GetXaxis().FindBin(x_center)
            new_iy = h_out.GetYaxis().FindBin(y_center)

            # Skip if outside range (under/overflow in ref)
            if (new_ix < 1 or new_ix > h_out.GetNbinsX() or
                new_iy < 1 or new_iy > h_out.GetNbinsY()):
                continue

            # Add content and combine errors in quadrature
            old_content = h_out.GetBinContent(new_ix, new_iy)
            old_error   = h_out.GetBinError(new_ix, new_iy)

            new_content = old_content + content
            new_error = (old_error**2 + error**2)**0.5

            h_out.SetBinContent(new_ix, new_iy, new_content)
            h_out.SetBinError(new_ix, new_iy, new_error)

    return h_out

def scale_histogram_xaxis(h_in, scale, name_suffix="_scaled"):
    """
    Return a new TH1 with the x-axis scaled by 'scale':
      x_new = scale * x_old

    Parameters
    ----------
    h_in : ROOT.TH1
        Input 1D histogram.
    scale : float
        Scale factor for the x axis.
    name_suffix : str, optional
        Suffix appended to the cloned histogram name.

    Returns
    -------
    ROOT.TH1
        New histogram with scaled x-axis and re-distributed contents.
    """
    if scale == 0:
        raise ValueError("Scale factor must be nonzero.")

    # Get original properties
    nbins = h_in.GetNbinsX()
    xaxis = h_in.GetXaxis()
    xmin  = xaxis.GetXmin()
    xmax  = xaxis.GetXmax()

    # Create a new histogram with scaled range and same binning
    hname = h_in.GetName() + name_suffix + "_" + str(scale)
    htitle = h_in.GetTitle() + f" (x scaled by {scale})"
    h_out = r.TH1F(hname, htitle, nbins, xmin * scale, xmax * scale)
    h_out.Sumw2()  # enable proper error handling

    # Loop over bins (1..nbins, skipping under/overflow here)
    for ibin in range(1, nbins + 1):
        x_center = xaxis.GetBinCenter(ibin)
        new_x = x_center * scale

        content = h_in.GetBinContent(ibin)
        error   = h_in.GetBinError(ibin)

        if content == 0 and error == 0:
            continue

        # Find corresponding bin in the new histogram
        jbin = h_out.FindBin(new_x)

        # Add content
        old_content = h_out.GetBinContent(jbin)
        h_out.SetBinContent(jbin, old_content + content)

        # Combine errors in quadrature
        old_error = h_out.GetBinError(jbin)
        h_out.SetBinError(jbin, np.sqrt(old_error**2 + error**2))

    # Optionally: copy entries, etc.
    h_out.SetEntries(h_in.GetEntries())

    return h_out

def P_0(IR, ITSROFlength=defaultITSROFlength, interactionRateFactor=1.):
    # Probability of 0 collisions in one ITS ROF
    # IR (Hz), ITSROFlength (s)
    # Not normalized, just returns the probability as a fraction
    return np.array(np.exp(-IR * interactionRateFactor * ITSROFlength))

# TODO: The "reference" VtxNContrib histogram, which we integrate to find p_ncontribs_leq, should in principle be the "true" VtxNContrib distribution.
#       Test how big an effect it is to fix this histogram to e.g. a well-behaved low IR run.

def prob_winner(n_contrib, n_lost, vtx_n_contrib_hist, ir, its_rof_length=defaultITSROFlength):
    """
    Given a UPC interaction rate IR and a distribution of VtxNContrib,
    calculate the probability that an event with n_contrib vertex contributors caused
    n_lost other events to not be reconstructed due to ITSROF pileup
    """
    if n_contrib < 16:
        n_contrib_bin = vtx_n_contrib_hist.GetXaxis().FindBin(n_contrib)
    else:
        n_contrib_bin = vtx_n_contrib_hist.GetXaxis().FindBin(16) # Events with n_contrib>=16 can only win over events with n_contrib<16. Otherwise, the competitor survives
    p_ncontribs_leq = vtx_n_contrib_hist.Integral(0, n_contrib_bin) / vtx_n_contrib_hist.Integral()
    p_n_pileup = poisson.pmf(n_lost, ir * its_rof_length)
    return p_n_pileup * (p_ncontribs_leq ** n_lost), p_ncontribs_leq


def get_n_lost_events(vtx_n_contrib_hist, hadronic_ir, upc_ir_factor, its_rof_length=defaultITSROFlength, min_num=1e-4, vtx_n_contrib_hist_reference=None, do_print=False):
    """
    Returns a list of number of lost events for each bin in the vtx_n_contrib_hist
    """
    if vtx_n_contrib_hist_reference is None:
        vtx_n_contrib_hist_reference = vtx_n_contrib_hist
    upc_ir = hadronic_ir * upc_ir_factor
    list_n_lost = []
    for i in range(1, vtx_n_contrib_hist.GetNbinsX() + 1):
        n_contrib = int(vtx_n_contrib_hist.GetBinLowEdge(i))
        k = 1
        term = vtx_n_contrib_hist.GetBinContent(i) * prob_winner(n_contrib, k, vtx_n_contrib_hist_reference, upc_ir)[0] # Probability that we lost 1 event 
        sum_terms = 0
        while (term > min_num): # Stop when next term gives negligible event loss
            sum_terms += term
            k += 1
            term = vtx_n_contrib_hist.GetBinContent(i) * k * prob_winner(n_contrib, k, vtx_n_contrib_hist_reference, upc_ir)[0] 
        list_n_lost.append(sum_terms)
        if do_print:
            print(f"At bin {i}, VtxNcontrib={n_contrib} we have {vtx_n_contrib_hist.GetBinContent(i)} events, and lost {list_n_lost[-1]} events.")
            print(f"  The expected event loss per visible event is {sum_terms / vtx_n_contrib_hist.GetBinContent(i) if sum_terms > 0 else 0}, calculated using P(n <= {n_contrib}) = {prob_winner(n_contrib, k, vtx_n_contrib_hist_reference, upc_ir)[1]}") 
            print(f"  Used {k-1} terms in the series")
    return list_n_lost

def collect_instantaneous_ir(run_list, df, interval=60):
    """
    Collect interaction rate as a function of time by polling the CCDB.
    run_list: List of run numbers
    df: Dataframe containing SOR and EOR timestamps for the run
    interval: Timestamp interval in seconds used for CCDB fetching
    """
    r.gSystem.Load("libO2CCDB.so")
    r.gInterpreter.Declare('#include "ctpRateFetcher.h"')
    
    for run in run_list:
        fetcher = r.o2.ctpRateFetcher()
        ccdb_manager = r.o2.ccdb.BasicCCDBManager.instance()
        # Configure CCDB manager, mirroring the DQ table-reader
        ccdb_manager.setURL("http://alice-ccdb.cern.ch")
        ccdb_manager.setCaching(True)
        ccdb_manager.setLocalObjectValidityChecking()
        print(f"--- Working on run {run} ---")
        run_number = int(run)
        sor = int(df.loc[run, "Start timestamp TRG"]) # Epoch time in ms
        eor = int(df.loc[run, "End timestamp TRG"]) # Epoch time in ms
        print(f"Fetching IR in run {run_number} from timestamp {sor} to {eor} in {interval} second intervals")
        ts = []
        irs = []
        t = sor
        while (t < eor):
            ts.append(t)
            irs.append(fetcher.fetch(ccdb_manager, t, run_number, "ZNC hadronic"))
            t += 1000 * interval
        # Make sure to get the final point
        ts.append(eor)
        irs.append(fetcher.fetch(ccdb_manager, eor, run_number, "ZNC hadronic"))
        # Format and save
        ts = np.array(ts)
        irs = np.array(irs)
        data = np.column_stack((ts, irs))
        np.savetxt(f"/home/sigurd/cernbox/notebooks/pyD0/interactionRates/{run_number}.txt", data, fmt="%i %.6e", header="timestamp_ms interaction_rate")

def integrate_between_ys(xs, ys, y0, y1):
    """
    Integrate the function ys along xs in every region where y0 <= y <= y1
    Values between datapoints are linearly interpolated
    Returns:
    Integral,
    Array of x values where y crossed y0 or y1,
    Array of y values where y crossed y0 or y1,
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if np.any(np.diff(xs) <= 0):
        raise ValueError("xs must be strictly increasing")

    total = 0.0
    x_cross = []
    y_cross = []

    # Track whether the global first/last point is used in the integral
    uses_first = False
    uses_last = False

    n = len(xs)

    for i in range(n - 1):
        x0, x1 = xs[i], xs[i+1]
        y_start, y_end = ys[i], ys[i+1]

        seg_min = min(y_start, y_end)
        seg_max = max(y_start, y_end)

        # If segment doesn't intersect [y0, y1] at all, skip it
        if seg_max <= y0 or seg_min >= y1:
            continue

        # Local containers (segment including interpolation points)
        x_local = [x0]
        y_local = [y_start]

        # Linear interpolation helper: x at given y on this segment
        if y_end != y_start:
            m = (x1 - x0) / (y_end - y_start)  # dx/dy
            def x_at_y(y):
                return x0 + (y - y_start) * m
        else:
            m = None

        # Crossings with y0 and y1
        for y_level in (y0, y1):
            if (y_level - y_start) * (y_level - y_end) < 0 and m is not None:
                xc = x_at_y(y_level)
                x_local.append(xc)
                y_local.append(y_level)
                x_cross.append(xc)
                y_cross.append(y_level)

        x_local.append(x1)
        y_local.append(y_end)

        # Sort along x
        x_local = np.array(x_local)
        y_local = np.array(y_local)
        order = np.argsort(x_local)
        x_local = x_local[order]
        y_local = y_local[order]

        # Integrate only sub‑segments fully inside [y0, y1]
        for j in range(len(x_local) - 1):
            xl0, xl1 = x_local[j],   x_local[j+1]
            yl0, yl1 = y_local[j],   y_local[j+1]

            if (y0 <= yl0 <= y1) and (y0 <= yl1 <= y1):
                total += np.trapezoid([yl0, yl1], x=[xl0, xl1])

                # Mark use of global endpoints if this sub‑segment touches them
                if i == 0 and np.isclose(xl0, xs[0]):
                    uses_first = True
                if i+1 == n-1 and np.isclose(xl1, xs[-1]):
                    uses_last = True

    # If the integration reaches the first/last point, store them as "crossings"
    if uses_first:
        x_cross.insert(0, xs[0])
        y_cross.insert(0, ys[0])
    if uses_last:
        x_cross.append(xs[-1])
        y_cross.append(ys[-1])

    return total, np.array(x_cross), np.array(y_cross)

def get_lumi_vs_ir(run_list, ir_bin_edges, lumi_path="/media/sigurd/T7/analysis/data/LHC23_PbPb_pass5_train590144/mergedAnalysisResults_good.root", do_print=False):
    """
    ir_bin_edges: kHz
    """
    if do_print:
        print(f"Processing {len(run_list)} runs...")
    lumi_file = r.TFile.Open("/media/sigurd/T7/analysis/data/LHC23_PbPb_pass5_train590144/mergedAnalysisResults_good.root")
    lumi_hist_znc = lumi_file.Get('eventselection-run3').Get('luminosity').Get('hLumiZNCafterBCcuts')
    hist = r.TH1D('lumi_ir', 'lumi_ir', len(ir_bin_edges)-1, np.asarray(ir_bin_edges, 'd'))
    hist.SetDirectory(0)
    hist.GetXaxis().SetTitle('ZNC hadronic interaction rate (Hz)')
    hist.GetYaxis().SetTitle('ZNC luminosity (1/µb)')
    for i, (ir_0, ir_1) in enumerate(zip(ir_bin_edges[:-1], ir_bin_edges[1:])):
        if do_print:
            print(f"--- {ir_0:.3f} < IR < {ir_1:.3f} kHz ---")
        L = 0
        n_contributing_runs = 0
        for run in run_list:
            data = np.genfromtxt(f"/home/sigurd/cernbox/notebooks/pyD0/interactionRates/{run}.txt", names=True)
            ts = data['timestamp_ms']
            ts = (ts - ts[0])/1000 # Convert to seconds from SOR
            irs = data['interaction_rate']
            irs = irs/1000 # Convert from Hz to kHz
            
            integrated_lumi = lumi_hist_znc.GetBinContent(lumi_hist_znc.GetXaxis().FindBin(run))
            factor = integrated_lumi / np.trapezoid(irs, ts)
            lumis = factor * irs
            
            N, ts_crossings, ir_crossings = integrate_between_ys(ts, irs, ir_0, ir_1)
            tmp_L = factor * N
            if tmp_L > 0:
                n_contributing_runs += 1
                if do_print:
                    print(f"  Run {run} has {tmp_L:.3f} 1/µb of luminosity between {ir_0} kHz and {ir_1} kHz")
            L += tmp_L

        if do_print:
            print(f"  {n_contributing_runs} runs contributed in total {L:.3f} 1/µb of luminosity between {ir_0} kHz and {ir_1} kHz")
        hist.SetBinContent(i+1, L)
    return hist

def plot_run_by_run_ir_lumi(run_list, ir_0, ir_1, lumi_path="/media/sigurd/T7/analysis/data/LHC23_PbPb_pass5_train590144/mergedAnalysisResults_good.root", do_print=False, y_axis='lumi', ncols=5, scale=1.0):
    """
    For every run in run_list, illustrate the contribution to the luminosity between interaction rate ir_0 to ir_1
    """
    lumi_file = r.TFile.Open("/media/sigurd/T7/analysis/data/LHC23_PbPb_pass5_train590144/mergedAnalysisResults_good.root")
    lumi_hist_znc = lumi_file.Get('eventselection-run3').Get('luminosity').Get('hLumiZNCafterBCcuts')
    total_lumi = lumi_hist_znc.Integral()

    nrows = int(np.ceil(len(run_list)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(scale*21*ncols/5, scale*3*nrows), squeeze=False)
    run_list.sort() # Plot in ascending order of run number
    Ls = np.zeros_like(run_list, dtype=float)
    for i, run in enumerate(run_list):
        row = int(np.floor(i/ncols))
        col = i%ncols
        ax = axes[row, col]

        data = np.genfromtxt(f"/home/sigurd/cernbox/notebooks/pyD0/interactionRates/{run}.txt", names=True)
        ts = data['timestamp_ms']
        ts = (ts - ts[0])/1000 # Convert to seconds from SOR
        irs = data['interaction_rate']
        irs = irs/1000 # Convert Hz to kHz

        N, ts_crossings, ir_crossings = integrate_between_ys(ts, irs, ir_0, ir_1)

        integrated_lumi = lumi_hist_znc.GetBinContent(lumi_hist_znc.GetXaxis().FindBin(run))
        factor = integrated_lumi / np.trapezoid(irs, ts)
        lumis = factor * irs
        if y_axis == 'lumi':
            ax.plot(ts, lumis*1000, '-')
            ax.set_ylabel("ZNC instant. lumi. (mb$^{-1}$/s)")
            ax.axhline(ir_0 * factor * 1000, ts[0], ts[-1], color='green', linestyle='--', alpha=0.5)
            ax.axhline(ir_1 * factor * 1000, ts[0], ts[-1], color='green', linestyle='--', alpha=0.5)
        elif y_axis == 'ir':
            ax.plot(ts, irs, '-')
            ax.set_ylabel("ZNC hadronic interaction rate (kHz)")
            ax.axhline(ir_0, ts[0], ts[-1], color='green', linestyle='--', alpha=0.5)
            ax.axhline(ir_1, ts[0], ts[-1], color='green', linestyle='--', alpha=0.5)
        else:
            raise Exception(f"Don't know how to interpret y_axis={y_axis}!")
        ax.set_xlabel("Time from SOR (s)") # Convert 1/µb -> 1/mb
        ax.set_title(run)
        ax.text(0.05, 0.05, f"Integral = {np.trapezoid(lumis, ts):.2f} 1/µb", ha='left', va='bottom', transform=ax.transAxes)

        if do_print:
            print(f"Found {len(ts_crossings)/2} regions:")
        for t0, t1 in zip(ts_crossings[::2], ts_crossings[1::2]):
            if do_print:
                print(f"{t0} -> {t1}")
            ax.axvspan(t0, t1, alpha=0.5, color='green', lw=0)
        L = factor * N
        Ls[i] = L
        if do_print:
            print(f"Run {run} has {L:.3f} 1/µb of luminosity between {ir_0} kHz and {ir_1} kHz")
        ax.text(0.05, 0.12, f"Contribution = {L:.2f} 1/µb", ha='left', va='bottom', transform=ax.transAxes)

    fig.tight_layout()

    # Now that we have the list of contributions, order the plots in descending order of contributing lumi
    idx = np.argsort(Ls)[::-1]
    axes_flattened = axes.ravel()
    positions = [ax.get_position() for ax in axes_flattened]
    axes_reordered = axes_flattened[idx]
    for k, old_i in enumerate(idx):
        axes_flattened[old_i].set_position(positions[k])

    print(f"Total lumi between between {ir_0:.2f} kHz and {ir_1:.2f} kHz: {Ls.sum():.2f} 1/µb ({100*Ls.sum()/total_lumi:.2f}% of the total {total_lumi:.2f} 1/µb in the runlist)")
    plt.show()

def root_to_hist(hroot):
    nbins = hroot.GetNbinsX()
    edges = np.array([hroot.GetBinLowEdge(i) for i in range(1, nbins + 2)])
    counts = np.array([hroot.GetBinContent(i) for i in range(1, nbins + 1)])
    errors = np.array([hroot.GetBinError(i) for i in range(1, nbins + 1)])

    h = hist.Hist(
        hist.axis.Variable(edges, name=hroot.GetXaxis().GetName(), label=hroot.GetXaxis().GetTitle()),
        storage=hist.storage.Weight()
    )

    view = h.view()
    view.value[...] = counts
    view.variance[...] = errors**2

    return h

def equal_stat_y_slices(h2, n_slices, start_bin=1, end_bin=None):
    """
    h2: TH2 (e.g. TH2F)
    n_slices: desired number of Y slices with equal total entries (over X)
    start_bin: first Y bin to include (1-based, inclusive, no under/overflow)
    end_bin:   last Y bin to include (1-based, inclusive, no under/overflow).
               If None, defaults to nbins_y.

    Returns:
        A list of (ybin_lo, ybin_hi) tuples (1-based, inclusive),
        restricted to Y bins [start_bin .. end_bin].
    """
    nbins_x = h2.GetNbinsX()
    nbins_y = h2.GetNbinsY()

    if end_bin is None:
        end_bin = nbins_y

    # Sanity checks
    if start_bin < 1 or start_bin > nbins_y:
        raise ValueError(f"start_bin={start_bin} out of range [1, {nbins_y}]")
    if end_bin < 1 or end_bin > nbins_y:
        raise ValueError(f"end_bin={end_bin} out of range [1, {nbins_y}]")
    if start_bin > end_bin:
        raise ValueError(f"start_bin={start_bin} must be <= end_bin={end_bin}")

    # 1) Sum over X for each Y bin from start_bin .. end_bin
    per_y = []
    total = 0.0
    for j in range(start_bin, end_bin + 1):
        s = 0.0
        for i in range(1, nbins_x + 1):
            s += h2.GetBinContent(i, j)
        per_y.append(s)
        total += s

    if total <= 0:
        raise RuntimeError(
            f"Histogram has no entries in Y bins [{start_bin}..{end_bin}]."
        )

    # 2) Cumulative along Y within [start_bin..end_bin]
    cumulative = []
    run = 0.0
    for s in per_y:
        run += s
        cumulative.append(run)

    target_per_slice = total / float(n_slices)

    # 3) Find upper-bin edge for each slice (except the last)
    upper_edges = []  # Y-bin indices (1..nbins_y) as upper edges
    k = 1  # we look for k * target_per_slice
    for idx in range(len(per_y)):          # idx is 0..(end_bin-start_bin)
        j = start_bin + idx                # real bin number
        if k >= n_slices:
            break
        if cumulative[idx] >= k * target_per_slice:
            upper_edges.append(j)
            k += 1

    # At most n_slices-1 boundaries; last boundary forced at end_bin
    upper_edges = upper_edges[:n_slices-1]
    upper_edges.append(end_bin)

    # 4) Convert upper edges into (lo, hi) bin ranges
    slices = []
    prev_hi = start_bin - 1
    for hi in upper_edges:
        lo = prev_hi + 1
        slices.append((lo, hi))
        prev_hi = hi

    return slices

def trim_trailing_zeros(h_ref, *h_others, do_print=True):
    """
    Remove bins from scikit-hep/hist histograms based on trailing zeros in h_ref
    """
    vals = h_ref.values()
    nonzero = vals != 0

    if not nonzero.any():
        # nothing non-zero; return original histograms unchanged or empty
        return (h_ref,) + h_others

    last_nonzero_idx = int(np.where(nonzero)[0][-1])
    mask = np.arange(len(vals)) <= last_nonzero_idx

    # print(h_ref[:last_nonzero_idx+1])
    h_ref_trimmed = h_ref[0:last_nonzero_idx+1]
    h_others_trimmed = tuple(h[0:last_nonzero_idx+1] for h in h_others)
    n_trimmed = len(mask)-np.sum(mask)
    if do_print:
        print(f"Removed {n_trimmed} bins based on trailing zeros in {h_ref.axes[0].name}")
    return (h_ref_trimmed,) + h_others_trimmed, n_trimmed

def trim_trailing_zeros_root(h_ref, *h_others, do_print=True):
    """
    Trim trailing zero bins from a reference 1D ROOT histogram (TH1*),
    and apply the same trimming to other histograms with identical binning.

    Parameters
    ----------
    h_ref : ROOT.TH1
        Reference histogram; trailing zeros are determined from this one.
    *h_others : ROOT.TH1
        Other histograms to trim using the same bin range.

    Returns
    -------
    tuple of ROOT.TH1
        (h_ref_trimmed, h_other1_trimmed, ...)

    Notes
    -----
    - Only works for 1D histograms.
    - Assumes all histograms have the same number of bins and bin edges.
    """
    nbins = h_ref.GetNbinsX()

    # Find last non-zero bin (ignore underflow bin 0 and overflow nbins+1)
    last_nonzero = 0
    for i in range(1, nbins + 1):
        if h_ref.GetBinContent(i) != 0:
            last_nonzero = i

    if last_nonzero == 0:
        # All bins are zero: return original histograms unchanged
        print(f"All bins in {h_ref.GetName()} are zero")
        n_trimmed = 0
        return (h_ref,) + h_others + (n_trimmed,)

    n_trimmed = nbins - last_nonzero

    # New number of bins is last_nonzero; keep same axis range
    xmin = h_ref.GetXaxis().GetXmin()
    xmax = h_ref.GetBinLowEdge(last_nonzero + 1)  # upper edge of last kept bin

    def _trim_one(h, name_suffix="_trimmed"):
        # Create a new histogram with fewer bins and same range
        name = h.GetName() + name_suffix
        title = h.GetTitle()
        h_trim = r.TH1D(name, title, last_nonzero, xmin, xmax)

        # Copy contents (and errors)
        for i in range(1, last_nonzero + 1):
            content = h.GetBinContent(i)
            h_trim.SetBinContent(i, content)
            err = h.GetBinError(i)
            h_trim.SetBinError(i, err)

        return h_trim

    h_ref_trim = _trim_one(h_ref)
    h_others_trim = tuple(_trim_one(h) for h in h_others)

    if do_print:
        print(f"Removed {n_trimmed} bins based on trailing zeros in {h_ref.GetName()}")
    return (h_ref_trim,) + h_others_trim + (n_trimmed,)

def add_graphs(g1, g2, name="g_sum", title=""):
    n1 = g1.GetN()
    n2 = g2.GetN()
    if n1 != n2:
        raise ValueError("Graphs have different number of points")

    g_sum = r.TGraphAsymmErrors(n1)
    g_sum.SetName(name)
    g_sum.SetTitle(title)

    for i in range(n1):
        x1 = g1.GetX()[i]
        y1 = g1.GetY()[i]
        exl1 = g1.GetErrorXlow(i)
        exh1 = g1.GetErrorXhigh(i)
        eyl1 = g1.GetErrorYlow(i)
        eyh1 = g1.GetErrorYhigh(i)

        x2 = g2.GetX()[i]
        y2 = g2.GetY()[i]
        exl2 = g2.GetErrorXlow(i)
        exh2 = g2.GetErrorXhigh(i)
        eyl2 = g2.GetErrorYlow(i)
        eyh2 = g2.GetErrorYhigh(i)

        # sanity: same x (within tolerance)
        if abs(x1 - x2) > 1e-9:
            raise ValueError(f"x mismatch at point {i}: {x1} vs {x2}")

        x  = x1
        y  = y1 + y2

        # choose x errors (here: take the max; or take from g1 if they’re identical)
        exl = max(exl1, exl2)
        exh = max(exh1, exh2)

        # combine y errors in quadrature (assume uncorrelated)
        eyl = np.sqrt(eyl1**2 + eyl2**2)
        eyh = np.sqrt(eyh1**2 + eyh2**2)

        g_sum.SetPoint(i, x, y)
        g_sum.SetPointError(i, exl, exh, eyl, eyh)

    return g_sum

def set_alice_style():
    r.gStyle.Reset("Plain");
    r.gStyle.SetOptTitle(0);
    r.gStyle.SetOptStat(0);
    r.gStyle.SetPalette(1);
    r.gStyle.SetCanvasColor(10)
    r.gStyle.SetCanvasBorderMode(0);
    r.gStyle.SetFrameLineWidth(1);
    r.gStyle.SetFrameFillColor(r.kWhite);
    r.gStyle.SetPadColor(10);
    r.gStyle.SetPadTickX(1);
    r.gStyle.SetPadTickY(1);
    r.gStyle.SetPadBottomMargin(0.15);
    r.gStyle.SetPadLeftMargin(0.15);
    r.gStyle.SetHistLineWidth(1);
    r.gStyle.SetHistLineColor(r.kRed);
    r.gStyle.SetFuncWidth(2);
    r.gStyle.SetFuncColor(r.kGreen);
    r.gStyle.SetLineWidth(2);
    r.gStyle.SetLabelSize(0.045,"xyz");
    r.gStyle.SetLabelOffset(0.01,"y");
    r.gStyle.SetLabelOffset(0.01,"x");
    r.gStyle.SetLabelColor(r.kBlack,"xyz");
    r.gStyle.SetTitleSize(0.05,"xyz");
    r.gStyle.SetTitleOffset(1.25,"y");
    r.gStyle.SetTitleOffset(1.25,"x");
    r.gStyle.SetTitleFillColor(r.kWhite);
    r.gStyle.SetTextSizePixels(26);
    r.gStyle.SetTextFont(42);
    r.gStyle.SetLegendBorderSize(0);
    r.gStyle.SetLegendFillColor(r.kWhite);
    r.gStyle.SetLegendFont(42);

def create_seaborn_palette(name="bright"):
    rgb_list = sns.color_palette(name)
    colors = []
    base_index = 10000
    for i, rgb in enumerate(rgb_list):
        c = r.TColor(rgb[0], rgb[1], rgb[2])
        colors.append(c)
    return colors

def create_zdc_migration_matrix(ea, ec, pa, pc):
    """
    Returns the ZDC migration matrix C, where
    N = CM,
    N is the vector of observed counts, M is the vector of true counts
    The row ordering is 0n0n, 0nXn, Xn0n, XnXn
    Input: 
    ea : ZNA efficiency
    ec : ZNC efficiency
    pa : ZNA pileup probability
    pc : ZNC pileup probability
    """
    row_0n0n = [1 - pa*(1-pc) - pc*(1-pa) - pa*pc, (1-ec)*(1-pa)*(1-pc), (1-ea)*(1-pa)*(1-pc), (1-ea)*(1-ec)*(1-pa)*(1-pc)]
    row_0nXn = [pc*(1-pa), 1 - ( (1-ec)*(1-pa)*(1-pc) + pa ), pc*(1-pa)*(1-ea), (1-ea)*(1-pa)*(ec + (1-ec)*pc)]
    row_Xn0n = [pa*(1-pc), pa*(1-pc)*(1-ec), 1 - ( (1-ea)*(1-pc)*(1-pa) + pc ), (1-ec)*(1-pc)*(ea + (1-ea)*pa)]
    row_XnXn = [pa*pc, pa*(ec + (1-ec)*pc), pc*(ea + (1-ea)*pa), 1 - ( (1-ea)*(1-pa)*(ec+(1-ec)*pc) + (1-ec)*(1-pc)*(ea+(1-ea)*pa) + (1-ea)*(1-ec)*(1-pa)*(1-pc))]
    return np.array([row_0n0n, row_0nXn, row_Xn0n, row_XnXn])

def get_mass_pt_in_neutron_classes(h):
    """
    From a 4-dim histogram of mass, pt, ZNA time, ZNC time,
    obtain the mass vs pt histograms in the 4 different neutron classes
    Assumes the following binning in ZNA and ZNC time: [-6, -2, 2, 6]
    """
    # Get the pieces of the 0n0n histogram
    h.GetAxis(2).SetRange(0,1)
    h.GetAxis(3).SetRange(0,1)
    proj0n0n_ll = h.Projection(1, 0)
    proj0n0n_ll.SetName("0n0n_ll")
    h.GetAxis(2).SetRange(0,1)
    h.GetAxis(3).SetRange(3,4)
    proj0n0n_ul = h.Projection(1, 0)
    proj0n0n_ul.SetName("0n0n_ul")
    h.GetAxis(2).SetRange(3,4)
    h.GetAxis(3).SetRange(0,1)
    proj0n0n_lr = h.Projection(1, 0)
    proj0n0n_lr.SetName("0n0n_lr")
    h.GetAxis(2).SetRange(3,4)
    h.GetAxis(3).SetRange(3,4)
    proj0n0n_ur = h.Projection(1, 0)
    proj0n0n_ur.SetName("0n0n_ur")
    # Add together the pieces of the 0n0n histogram
    proj0n0n = proj0n0n_ll.Clone("0n0n")
    proj0n0n.Reset()
    proj0n0n.Add(proj0n0n_ll)
    proj0n0n.Add(proj0n0n_ul)
    proj0n0n.Add(proj0n0n_lr)
    proj0n0n.Add(proj0n0n_ur)
    print(f"0n0n: {proj0n0n.GetEntries()} entries")

    # Get the pieces of the 0nXn histogram
    h.GetAxis(2).SetRange(0,1)
    h.GetAxis(3).SetRange(2,2)
    proj0nXn_l = h.Projection(1, 0)
    proj0nXn_l.SetName("0nXn_l")
    h.GetAxis(2).SetRange(3,4)
    h.GetAxis(3).SetRange(2,2)
    proj0nXn_r = h.Projection(1, 0)
    proj0nXn_r.SetName("0nXn_r")
    # Add together the pieces of the 0nXn histogram
    proj0nXn = proj0nXn_l.Clone("0nXn")
    proj0nXn.Reset()
    proj0nXn.Add(proj0nXn_l)
    proj0nXn.Add(proj0nXn_r)
    print(f"0nXn: {proj0nXn.GetEntries()} entries")

    # Get the pieces of the Xn0n histogram
    h.GetAxis(2).SetRange(2,2)
    h.GetAxis(3).SetRange(0,1)
    projXn0n_l = h.Projection(1, 0)
    projXn0n_l.SetName("Xn0n_l")
    h.GetAxis(2).SetRange(2,2)
    h.GetAxis(3).SetRange(3,4)
    projXn0n_u = h.Projection(1, 0)
    projXn0n_u.SetName("Xn0n_r")
    projXn0n = projXn0n_l.Clone("Xn0n")
    projXn0n.Reset()
    projXn0n.Add(projXn0n_l)
    projXn0n.Add(projXn0n_u)
    print(f"Xn0n: {projXn0n.GetEntries()} entries")

    # Get the XnXn histogram
    h.GetAxis(2).SetRange(2,2)
    h.GetAxis(3).SetRange(2,2)
    projXnXn = h.Projection(1, 0)
    projXnXn.SetName("XnXn")
    print(f"XnXn: {projXnXn.GetEntries()} entries")

    projs = [proj0n0n, proj0nXn, projXn0n, projXnXn]
    return projs

def make_ratio_canvas(h_num, h_den, rho=0.0, cname="c_ratio", log=True, 
                      name_num=None, name_den=None, shortname_num='Num', shortname_den='Den'):
    """
    Create a tratioPlot-like canvas for two histograms with correlated uncertainties.

    Parameters
    ----------
    h_num : ROOT.TH1
        Numerator histogram (e.g. data).
    h_den : ROOT.TH1
        Denominator histogram (e.g. MC / prediction).
    rho : float
        Correlation coefficient between the *total uncertainties* of the two histograms.
        Range typically [-1, 1].
    cname : str
        Name of the output TCanvas.

    Returns
    -------
    ROOT.TCanvas
        Canvas with upper pad (histograms) and lower pad (ratio).
    ROOT.TH1
        Ratio histogram with propagated errors.
    """

    # Basic checks
    nbins_num = h_num.GetNbinsX()
    nbins_den = h_den.GetNbinsX()
    if nbins_num != nbins_den:
        raise ValueError("Histograms must have the same number of bins.")

    palette = create_seaborn_palette()
    color_num = palette[1].GetNumber()
    color_den = palette[2].GetNumber()

    # --- 1. Build ratio histogram with custom error propagation ---
    ratio = h_num.Clone(h_num.GetName() + "_ratio")
    ratio.SetTitle("")
    ratio.Reset()  # We'll fill contents and errors manually

    for i in range(1, nbins_num + 1):
        A  = h_num.GetBinContent(i)
        B  = h_den.GetBinContent(i)
        eA = h_num.GetBinError(i)
        eB = h_den.GetBinError(i)

        if B == 0:
            ratio.SetBinContent(i, 0.0)
            ratio.SetBinError(i, 0.0)
            continue

        R = A / B
        ratio.SetBinContent(i, R)

        # Variances
        var_A = eA * eA
        var_B = eB * eB

        # Covariance using global correlation coefficient rho
        cov_AB = rho * np.sqrt(var_A * var_B)

        # Error propagation for R = A / B:
        # dr/dA =  1 / B
        # dr/dB = -A / B^2
        dA = 1.0 / B
        dB = -A / (B * B)

        var_r = (dA * dA) * var_A + (dB * dB) * var_B + 2.0 * dA * dB * cov_AB
        if var_r < 0:
            var_r = 0.0  # numerical safety

        ratio.SetBinError(i, np.sqrt(var_r))

    if 'c' in globals():
        del c
    c = r.TCanvas(cname, "Ratio Plot", 800, 800)
    c.Divide(1, 2)

    # Top pad (main histograms)
    pad1 = r.TPad("pad1", "pad1", 0.0, 0.30, 1.0, 1.0)
    pad1.SetBottomMargin(0.08)

    pad2 = r.TPad("pad2", "pad2", 0.0, 0.0, 1.0, 0.30)
    pad2.SetTopMargin(0.05)
    pad2.SetBottomMargin(0.25)

    pad1.Draw()
    pad2.Draw()

    pad1.cd()
    # Style for histograms
    h_num.SetMarkerStyle(20)
    h_num.SetMarkerColor(color_num)
    h_num.SetLineColor(color_num)

    h_den.SetMarkerStyle(21)
    h_den.SetMarkerColor(color_den)
    h_den.SetLineColor(color_den)

    # Determine drawing range
    h_num_max = h_num.GetMaximum()
    h_den_max = h_den.GetMaximum()
    max_y = 1.2 * max(h_num_max, h_den_max)
    h_num.SetMaximum(max_y)

    h_num.Draw("E1")
    h_den.Draw("E1 SAME")
    h_num.Draw("E1 SAME")

    leg = r.TLegend(0.65, 0.7, 0.89, 0.89)
    leg.SetBorderSize(0)
    if name_num is not None and name_den is not None:
        leg.AddEntry(h_num, name_num, "p")
        leg.AddEntry(h_den, name_den, "p")
        leg.Draw()
        pad1.Update()

    if log:
        pad1.SetLogy()

    pad2.cd()
    ratio.GetYaxis().SetTitle(f"{shortname_num} / {shortname_den}")
    ratio.GetYaxis().SetNdivisions(505)
    ratio.GetYaxis().SetTitleSize(0.10)
    ratio.GetYaxis().SetTitleOffset(0.4)
    ratio.GetYaxis().SetLabelSize(0.08)

    ratio.GetXaxis().SetTitle(h_num.GetXaxis().GetTitle())
    ratio.GetXaxis().SetTitleSize(0.10)
    ratio.GetXaxis().SetLabelSize(0.08)
    ratio.GetXaxis().SetLabelOffset(0.02)

    ratio.SetMarkerStyle(20)
    ratio.SetMarkerColor(r.kBlack)
    ratio.SetLineColor(r.kBlack)

    # Set sensible ratio range
    ratio.SetMinimum(0.5)
    ratio.SetMaximum(1.5)

    ratio.Draw("E1")

    # Reference line at 1
    xmin = h_num.GetXaxis().GetXmin()
    xmax = h_num.GetXaxis().GetXmax()
    line = r.TLine(xmin, 1.0, xmax, 1.0)
    line.SetLineStyle(2)
    line.SetLineColor(r.kGray+2)
    line.Draw("SAME")

    c.Draw()
    return c, ratio, leg, line

def uproot_open_fallback(paths):
    for p in paths:
        try:
            return uproot.open(p)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"None of {paths!r} exist")

def absolute_profileX(h2):
    """
    Return a TProfileX of |Y|.

    An intermediate TH2 is created by reflecting negative-Y bins onto
    positive Y. Its bin errors are set to Poisson errors, sqrt(content).
    """
    xaxis = h2.GetXaxis()
    yaxis = h2.GetYaxis()

    nx = xaxis.GetNbins()
    ny = yaxis.GetNbins()

    # Preserve the X-axis binning.
    xedges = array(
        "d",
        [xaxis.GetBinLowEdge(ix) for ix in range(1, nx + 1)]
        + [xaxis.GetBinUpEdge(nx)],
    )

    # Construct the positive-|Y| axis from all original bin boundaries.
    #
    # This is appropriate provided no Y bin straddles zero. Usually zero
    # is already a bin boundary for a signed observable.
    abs_y_edges = sorted({
        abs(yaxis.GetBinLowEdge(iy))
        for iy in range(1, ny + 1)
    } | {
        abs(yaxis.GetBinUpEdge(iy))
        for iy in range(1, ny + 1)
    })
    abs_y_edges = array("d", abs_y_edges)

    h_abs = r.TH2D(
        f"{h2.GetName()}_absY",
        f"{h2.GetTitle()};{xaxis.GetTitle()};|{yaxis.GetTitle()}|",
        nx, xedges,
        len(abs_y_edges) - 1, abs_y_edges,
    )
    h_abs.SetDirectory(0)
    h_abs.Sumw2()

    # Fold negative and positive Y bins into bins in |Y|.
    for ix in range(1, nx + 1):
        x = xaxis.GetBinCenter(ix)

        for iy in range(1, ny + 1):
            content = h2.GetBinContent(ix, iy)
            if content == 0.0:
                continue

            abs_y = abs(yaxis.GetBinCenter(iy))

            # FindBin is used instead of Fill so the histogram represents
            # counts in bins, not a set of weighted Fill calls.
            iy_abs = h_abs.GetYaxis().FindBin(abs_y)
            old_content = h_abs.GetBinContent(ix, iy_abs)

            h_abs.SetBinContent(ix, iy_abs, old_content + content)

    # Assign Poisson uncertainties after merging +/- Y bins.
    for ix in range(1, nx + 1):
        for iy in range(1, h_abs.GetNbinsY() + 1):
            content = h_abs.GetBinContent(ix, iy)

            # This assumes h2 contains non-negative event counts.
            h_abs.SetBinError(ix, iy, np.sqrt(content) if content > 0 else 0.0)

    profile = h_abs.ProfileX(
        f"{h2.GetName()}_absProfileX",
        1,
        h_abs.GetNbinsY(),
        "",
    )
    profile.SetDirectory(0)

    return profile

def profile_to_graph(profile, graphName, graphTitle):
    graph = r.TGraphErrors()
    graph.SetName(graphName)
    graph.SetTitle(graphTitle)

    point = 0

    for ibin in range(1, profile.GetNbinsX() + 1):
        # Skip bins without entries. This is usually important for TProfiles.
        if profile.GetBinEntries(ibin) <= 0:
            continue

        x = profile.GetBinCenter(ibin)
        y = profile.GetBinContent(ibin)

        # Standard choice: horizontal uncertainty = half bin width.
        ex = 0.5 * profile.GetBinWidth(ibin)

        # For a TProfile, this is the profile uncertainty according to
        # the profile error option currently set on the input profile.
        ey = profile.GetBinError(ibin)

        graph.SetPoint(point, x, y)
        graph.SetPointError(point, ex, ey)
        point += 1

    return graph

def default_fitting(analysis):
    """
    Shortcut to doing the invariant mass fits in the 11-bin analysis. Makes it easier to do systematic variations.
    """
    analysis.fit_inv_mass(0, fit_functions.pol2, fitRange=[1.7, 2.0], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.005, 0.02], rebin=5, doPrint=False, reflParLimits=[None, [0.0,0.0001], None, None, None, None])
    analysis.fit_inv_mass(1, fit_functions.pol2, fitRange=[1.68, 2.04], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.005, 0.05], rebin=5, doPrint=False)
    analysis.fit_inv_mass(2, fit_functions.pol2, fitRange=[1.68, 2.04], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.005, 0.05], rebin=5, doPrint=False)
    analysis.fit_inv_mass(3, fit_functions.pol3, fitRange=[1.68, 2.04], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.005, 0.05], rebin=5, doPrint=False, backgroundInitialParams=[-29e3, 165e3, -317e3, 205e3])
    analysis.fit_inv_mass(4, fit_functions.exp, fitRange=[1.68, 2.04], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.005, 0.05], rebin=5, useCorrBkgTemplate=True, doPrint=False)
    analysis.fit_inv_mass(5, fit_functions.exp, fitRange=[1.68, 2.04], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.005, 0.05], rebin=5, useCorrBkgTemplate=True, doPrint=False)
    analysis.fit_inv_mass(6, fit_functions.exp, fitRange=[1.68, 2.04], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.005, 0.05], rebin=5, useCorrBkgTemplate=True, doPrint=False)
    analysis.fit_inv_mass(7, fit_functions.exp, fitRange=[1.66, 2.06], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.005, 0.05], rebin=5, useCorrBkgTemplate=True, doPrint=False)#, backgroundInitialParams=[25000, -3.14])
    analysis.fit_inv_mass(8, fit_functions.exp, fitRange=[1.64, 2.1], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.005, 0.05], rebin=10, useCorrBkgTemplate=True, doPrint=False)
    analysis.fit_inv_mass(9, fit_functions.exp, fitRange=[1.62, 2.1], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.005, 0.05], rebin=10, useCorrBkgTemplate=True, doPrint=False)
    analysis.fit_inv_mass(10, fit_functions.exp, fitRange=[1.62, 2.1], signalMuRange=[1.8, 1.9], signalSigmaRange=[0.01, 0.05], rebin=10, useCorrBkgTemplate=True, doPrint=False)

def plot_relative_deviations(nominal, *variations, title="Relative deviations",
                             x_title=None, y_range=None, y_padding=0.40, correlation=1.0):
    """
    Plot (variation - nominal) / nominal for an arbitrary number of variations.

    Parameters
    ----------
    nominal : ROOT.TH1
        Nominal histogram.

    *variations : tuple(ROOT.TH1, str)
        Each variation must be given as:
            (variation_histogram, legend_label)

    title : str, optional
        Plot title.

    x_title : str, optional
        X-axis title. Defaults to the nominal histogram X-axis title.

    y_range : tuple(float, float), optional
        Explicit Y-axis range, e.g. (-0.2, 0.2). If omitted, a range is
        computed automatically such that all variation points are visible.

    y_padding : float, optional
        Fractional padding added above and below the automatically determined
        Y-axis range. For example, 0.10 means 10% padding.

    Returns
    -------
    ROOT.TCanvas
    """
    scale = 1.25

    if not variations:
        raise ValueError("Provide at least one variation histogram.")

    if nominal.GetDimension() != 1:
        raise ValueError("This function currently supports one-dimensional histograms only.")

    n_bins = nominal.GetNbinsX()

    canvas_name = str(randint(0,1e9))
    canvas = r.TCanvas(canvas_name, title, 500, 400)
    canvas.SetGridy()

    deviation_hists = []
    all_values = []

    # Build every deviation histogram before drawing, so that we can find
    # a Y-axis range covering all of them.
    for i, item in enumerate(variations):
        try:
            variation, label = item
        except (TypeError, ValueError):
            raise ValueError(
                "Each variation must be given as a tuple: (histogram, legend_label)"
            )

        if variation.GetNbinsX() != n_bins:
            raise ValueError(
                f"Variation '{label}' has {variation.GetNbinsX()} bins, "
                f"but nominal has {n_bins}."
            )

        deviation = variation.Clone(f"{variation.GetName()}_relative_deviation_{i}")
        deviation.SetDirectory(0)
        deviation.Reset("ICES")

        # Only use normal histogram bins for determining the plotted range.
        for bin_idx in range(1, n_bins + 1):
            nominal_value = nominal.GetBinContent(bin_idx)
            variation_value = variation.GetBinContent(bin_idx)
            nominal_error = nominal.GetBinError(bin_idx)
            variation_error = variation.GetBinError(bin_idx)

            if nominal_value != 0.0:
                value = (variation_value - nominal_value) / nominal_value
                correlation = min(nominal_value, variation_value) / np.sqrt(nominal_value*variation_value)
                print(correlation)
                error = value * np.sqrt((variation_error/variation_value)**2 + (nominal_error/nominal_value)**2 - 2*(correlation*variation_error*nominal_error/(nominal_value*variation_value)))
                all_values.append(value)
            else:
                value = 0.0

            deviation.SetBinContent(bin_idx, value)
            deviation.SetBinError(bin_idx, error)

        colors = create_seaborn_palette()
        color = colors[i].GetNumber()
        deviation.SetLineColor(color)
        deviation.SetMarkerColor(color)
        deviation.SetLineWidth(2)
        deviation.SetMarkerStyle(20 + (i % 10))
        deviation.SetMarkerSize(0.9)

        deviation.SetTitle(title)
        deviation.GetYaxis().SetTitle("(variation - nominal) / nominal")
        deviation.GetYaxis().SetTitleOffset(1.6)
        deviation.GetXaxis().SetTitle(
            x_title if x_title is not None else nominal.GetXaxis().GetTitle()
        )
        deviation.GetXaxis().SetTitleSize(deviation.GetYaxis().GetTitleSize()*scale)
        deviation.GetYaxis().SetTitleSize(deviation.GetYaxis().GetTitleSize()*scale)
        deviation.GetXaxis().SetLabelSize(deviation.GetYaxis().GetLabelSize()*scale)
        deviation.GetYaxis().SetLabelSize(deviation.GetYaxis().GetLabelSize()*scale)
        deviation.SetTitleSize(deviation.GetYaxis().GetTitleSize()*scale)
        canvas.SetLeftMargin(0.16)
        canvas.SetBottomMargin(0.16)

        deviation_hists.append((deviation, label))

    if not all_values:
        raise ValueError("All nominal bins are zero; cannot determine deviations.")

    legend = r.TLegend(0.45, 0.88-0.07*len(deviation_hists), 0.88, 0.88)

    # Determine a range that contains every point.
    if y_range is None:
        y_min = min(all_values)
        y_max = max(all_values)

        # Include zero in the displayed range for easier interpretation.
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)

        span = y_max - y_min

        # Handle the case where every point has exactly the same value.
        if span == 0.0:
            span = max(abs(y_min), 1.0)

        padding = y_padding * span
        y_range = (y_min - padding, y_max + padding)

    # Draw after the common Y-axis range has been established.
    for i, (deviation, label) in enumerate(deviation_hists):
        deviation.GetYaxis().SetRangeUser(*y_range)

        draw_option = "PE" if i == 0 else "PE SAME"
        deviation.Draw(draw_option)

        legend.AddEntry(deviation, label, "lp")

    x_min = nominal.GetXaxis().GetXmin()
    x_max = nominal.GetXaxis().GetXmax()

    zero_line = r.TLine(x_min, 0.0, x_max, 0.0)
    zero_line.SetLineColor(r.kBlack)
    zero_line.SetLineStyle(2)
    zero_line.SetLineWidth(2)
    zero_line.Draw("SAME")

    legend.Draw()
    canvas.RedrawAxis()
    canvas.Update()

    # Keep references alive in PyROOT.
    canvas._deviation_hists = [hist for hist, _ in deviation_hists]
    canvas._deviation_legend = legend
    canvas._deviation_zero_line = zero_line

    _root_callables.append(canvas)
    #return canvas

import math
import ROOT as r


def d0_mass_widths_vs_pt(
    h2,
    pt_edges,
    d0_mass=1.86483,
    seed_half_window=0.050,
    n_sigma=1.2,
    max_columns=3,
):
    """
    Project a 2D mass-vs-pt histogram into mass spectra for user-defined
    pT intervals, fit each D0 mass peak with a Gaussian, display all fits
    in a grid in Jupyter, and return the fitted Gaussian widths.

    Parameters
    ----------
    h2 : ROOT.TH2
        Input 2D histogram. X axis must be mass and Y axis must be pT.
    pt_edges : sequence of float
        pT bin edges in user coordinates, e.g. [0, 1, 2, 3, 5, 10].
    d0_mass : float
        Approximate D0 mass, in the same units as the X axis.
    seed_half_window : float
        Initial half-width around d0_mass for estimating the peak mean
        and Gaussian width.
    max_columns : int
        Maximum number of canvas columns. Defaults to 3.

    Returns
    -------
    list of float
        Gaussian widths from final fits, one per pT interval. Empty or
        failed fits return NaN.
    """

    if len(pt_edges) < 2:
        raise ValueError("pt_edges must contain at least two values.")

    if not h2.InheritsFrom("TH2"):
        raise TypeError("h2 must be a 2D ROOT histogram (TH2).")

    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    xaxis = h2.GetXaxis()
    yaxis = h2.GetYaxis()

    n_pt_bins = len(pt_edges) - 1
    n_columns = min(max_columns, n_pt_bins)
    n_rows = math.ceil(n_pt_bins / n_columns)

    # One canvas containing all pT-bin fits.
    canvas = r.TCanvas(
        "cD0MassFits",
        "D0 mass fits versus pT",
        450 * n_columns,
        400 * n_rows,
    )
    canvas.Divide(n_columns, n_rows)

    widths = []
    width_errors = []
    root_objects = [canvas]

    for i, (pt_low, pt_high) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
        if pt_high <= pt_low:
            raise ValueError(
                f"Invalid pT range [{pt_low}, {pt_high}]: "
                "upper edge must be greater than lower edge."
            )

        canvas.cd(i + 1)
        r.gPad.SetTicks(1, 1)

        # Select ROOT bins corresponding to user pT interval [pt_low, pt_high).
        ybin_low = max(1, yaxis.FindFixBin(pt_low))
        ybin_high = min(
            yaxis.GetNbins(),
            yaxis.FindFixBin(math.nextafter(pt_high, -math.inf)),
        )

        hmass = h2.ProjectionX(
            f"hMass_ptbin_{i}",
            ybin_low,
            ybin_high,
            "e",
        )
        hmass.SetDirectory(0)
        hmass.SetTitle(f"{pt_low:g} < p_{{T}} < {pt_high:g};Mass;Candidates")
        hmass.SetMarkerStyle(20)
        hmass.SetMarkerSize(0.65)

        if hmass.GetEntries() <= 0:
            hmass.Draw("E")

            label = r.TPaveText(0.20, 0.42, 0.80, 0.58, "NDC")
            label.SetFillStyle(0)
            label.SetBorderSize(0)
            label.AddText("Empty projection")
            label.Draw()

            widths.append(float("nan"))
            width_errors.append(float("nan"))
            root_objects.extend([hmass, label])
            continue

        # Initial fit near the nominal D0 mass, used to estimate mu and sigma.
        seed_low = max(xaxis.GetXmin(), d0_mass - seed_half_window)
        seed_high = min(xaxis.GetXmax(), d0_mass + seed_half_window)

        peak_bin = hmass.GetMaximumBin()
        peak_height = hmass.GetBinContent(peak_bin)

        f_seed = r.TF1(f"fSeed_ptbin_{i}", "gaus", seed_low, seed_high)
        f_seed.SetParameters(max(peak_height, 1.0), d0_mass, 0.010)

        hmass.Fit(f_seed, "RQ0")

        mean_seed = f_seed.GetParameter(1)
        sigma_seed = abs(f_seed.GetParameter(2))

        if not math.isfinite(mean_seed):
            mean_seed = d0_mass

        if not math.isfinite(sigma_seed) or sigma_seed <= 0:
            sigma_seed = 0.010

        # Final fit in the range mu +/- n_sigma.
        fit_low = max(xaxis.GetXmin(), mean_seed - n_sigma * sigma_seed)
        fit_high = min(xaxis.GetXmax(), mean_seed + n_sigma * sigma_seed)

        f_gaus = r.TF1(f"fD0_ptbin_{i}", "gaus", fit_low, fit_high)
        f_gaus.SetParameters(max(peak_height, 1.0), mean_seed, sigma_seed)
        f_gaus.SetLineColor(r.kRed + 1)
        f_gaus.SetLineWidth(2)

        fit_result = hmass.Fit(f_gaus, "RQS0")
        fit_status = fit_result.Status()

        width = (
            abs(f_gaus.GetParameter(2))
            if fit_status == 0
            else float("nan")
        )
        widths.append(width)

        width_error = (
            abs(fit_result.GetErrors()[2])
            if fit_status == 0
            else float("nan")
        )
        width_errors.append(width_error)

        hmass.Draw("E")
        f_gaus.Draw("SAME")

        chi2 = f_gaus.GetChisquare()
        ndf = f_gaus.GetNDF()
        chi2_ndf = chi2 / ndf if ndf > 0 else float("nan")

        info = r.TPaveText(0.52, 0.60, 0.88, 0.88, "NDC")
        info.SetFillStyle(0)
        info.SetBorderSize(0)
        info.SetTextAlign(12)
        info.SetTextSize(0.045)
        info.AddText(f"{pt_low:g} < p_{{T}} < {pt_high:g}")
        info.AddText(f"#mu = {f_gaus.GetParameter(1):.6f}")
        info.AddText(f"#sigma = {width:.6f}")
        info.AddText(f"#chi^{{2}}/ndf = {chi2_ndf:.2f}")
        info.AddText(f"status = {fit_status}")
        info.Draw()

        root_objects.extend([hmass, f_seed, f_gaus, info])

    canvas.Modified()
    canvas.Update()
    canvas.Draw()

    # Keep objects alive for PyROOT/Jupyter.
    d0_mass_widths_vs_pt._root_objects = root_objects

    return widths, width_errors

def make_observable_template(
    h3_data,
    h2_mc,
    low_pt,
    high_pt,
    mass_thresholds=(1.7, 1.8, 1.9, 2.0),
    signal_to_background=0.091,
    observable_label="Observable",
    name_prefix="observable_template",
    make_plots=True,
    log_y=True,
    binning_tolerance=1e-10,
    x_range_comparison=None
):
    """
    Build an observable template from interpolated data sidebands and MC.

    Expected histogram axes
    -----------------------
    h3_data : ROOT.TH3
        X axis: mass
        Y axis: pT
        Z axis: observable, for example Lxy, DCA, decay length, etc.

    h2_mc : ROOT.TH2
        X axis: pT
        Y axis: the same observable represented by the Z axis of h3_data.

    The final template is a normalized mixture of interpolated background
    and normalized MC signal, using the supplied signal-to-background ratio.

    Parameters
    ----------
    h3_data : ROOT.TH3
        Data histogram with mass, pT, and observable axes.

    h2_mc : ROOT.TH2
        MC histogram with pT and observable axes.

    low_pt : float
        Lower edge of the selected pT interval.

    high_pt : float
        Upper edge of the selected pT interval.

    mass_thresholds : tuple[float, float, float, float]
        Boundaries defining the lower sideband, signal region, and upper
        sideband:

        (lower_sideband_low,
         lower_sideband_high,
         upper_sideband_low,
         upper_sideband_high)

        The signal mass region is between the second and third values.

    signal_to_background : float
        Signal-to-background ratio, S/B, in the selected pT interval.

    observable_label : str
        Human-readable observable label used in plot axes.

    name_prefix : str
        Prefix for unique ROOT object names.

    make_plots : bool
        If True, create and draw diagnostic canvases.

    log_y : bool
        If True, use logarithmic Y axes in the plots.

    binning_tolerance : float
        Absolute tolerance for comparisons of histogram bin edges.

    Returns
    -------
    dict
        Dictionary containing all intermediate histograms, the final
        template, canvases, and legends.

    Raises
    ------
    ValueError
        If mass thresholds are invalid, the requested pT interval is invalid,
        a histogram cannot be normalized, or data and MC observable binnings
        are incompatible.
    """

    # ------------------------------------------------------------------
    # Input validation.
    # ------------------------------------------------------------------
    if len(mass_thresholds) != 4:
        raise ValueError(
            "mass_thresholds must contain exactly four values."
        )

    (
        lower_sideband_low,
        lower_sideband_high,
        upper_sideband_low,
        upper_sideband_high,
    ) = mass_thresholds

    if not (
        lower_sideband_low
        < lower_sideband_high
        < upper_sideband_low
        < upper_sideband_high
    ):
        raise ValueError(
            "Mass thresholds must satisfy: "
            "m0 < m1 < m2 < m3."
        )

    if low_pt >= high_pt:
        raise ValueError("low_pt must be smaller than high_pt.")

    if signal_to_background < 0:
        raise ValueError("signal_to_background must be non-negative.")

    # Convert S/B into fractions of the total template.
    signal_fraction = signal_to_background / (1.0 + signal_to_background)
    background_fraction = 1.0 / (1.0 + signal_to_background)

    unique_id = uuid.uuid4().hex[:10]

    def make_name(label):
        return f"{name_prefix}_{label}_{unique_id}"

    def detach(histogram):
        """Detach a histogram from the active ROOT directory."""
        if hasattr(histogram, "SetDirectory"):
            histogram.SetDirectory(0)
        return histogram

    def normalized_clone(histogram, label):
        """Create a detached clone normalized to unit integral."""
        integral = histogram.Integral()

        if integral <= 0:
            raise ValueError(
                f"Cannot normalize histogram '{histogram.GetName()}': "
                f"integral is {integral}."
            )

        output = detach(histogram.Clone(make_name(label)))
        output.Scale(1.0 / integral)

        return output

    def get_interval_bins(axis, low_value, high_value):
        """
        Return bins corresponding to the half-open interval
        [low_value, high_value).
        """
        first_bin = axis.FindBin(low_value)

        # Use the value immediately below high_value so a bin that starts
        # exactly at high_value is not included.
        last_bin = axis.FindBin(
            math.nextafter(high_value, -math.inf)
        )

        if last_bin < first_bin:
            raise ValueError(
                f"No valid bins found in interval "
                f"[{low_value}, {high_value})."
            )

        return first_bin, last_bin

    def get_histogram_binning(histogram):
        """Return all regular-bin edges, including the final upper edge."""
        axis = histogram.GetXaxis()

        edges = [
            axis.GetBinLowEdge(bin_index)
            for bin_index in range(1, histogram.GetNbinsX() + 1)
        ]
        edges.append(axis.GetBinUpEdge(histogram.GetNbinsX()))

        return edges

    def check_compatible_binning(histogram_a, histogram_b):
        """
        Raise ValueError unless two TH1 histograms have identical X binning.
        """
        n_bins_a = histogram_a.GetNbinsX()
        n_bins_b = histogram_b.GetNbinsX()

        axis_a = histogram_a.GetXaxis()
        axis_b = histogram_b.GetXaxis()

        if n_bins_a != n_bins_b:
            raise ValueError(
                "Incompatible observable binning between data background "
                "and MC signal.\n"
                f"Data background: {n_bins_a} bins over "
                f"[{axis_a.GetXmin()}, {axis_a.GetXmax()}]\n"
                f"MC signal:      {n_bins_b} bins over "
                f"[{axis_b.GetXmin()}, {axis_b.GetXmax()}]\n"
                "The histograms must have the same number of bins and "
                "identical bin edges. No rebinning was performed."
            )

        edges_a = get_histogram_binning(histogram_a)
        edges_b = get_histogram_binning(histogram_b)

        for edge_index, (edge_a, edge_b) in enumerate(
            zip(edges_a, edges_b)
        ):
            if not math.isclose(
                edge_a,
                edge_b,
                rel_tol=0.0,
                abs_tol=binning_tolerance,
            ):
                raise ValueError(
                    "Incompatible observable binning between data background "
                    "and MC signal.\n"
                    f"First mismatched edge index: {edge_index}\n"
                    f"Data background edge: {edge_a}\n"
                    f"MC signal edge:      {edge_b}\n"
                    f"Data background: {n_bins_a} bins over "
                    f"[{axis_a.GetXmin()}, {axis_a.GetXmax()}]\n"
                    f"MC signal:      {n_bins_b} bins over "
                    f"[{axis_b.GetXmin()}, {axis_b.GetXmax()}]\n"
                    "The histograms must have identical bin edges. "
                    "No rebinning was performed."
                )

    # ------------------------------------------------------------------
    # Select the requested pT interval in data and project to a
    # mass-vs-observable TH2.
    # ------------------------------------------------------------------
    data_pt_axis = h3_data.GetYaxis()

    original_data_pt_first_bin = data_pt_axis.GetFirst()
    original_data_pt_last_bin = data_pt_axis.GetLast()

    data_pt_first_bin, data_pt_last_bin = get_interval_bins(
        data_pt_axis,
        low_pt,
        high_pt,
    )

    data_pt_axis.SetRange(data_pt_first_bin, data_pt_last_bin)

    try:
        # For h3_data with X=mass, Y=pT, Z=observable:
        # Project3D("zx") has mass on X and observable on Y.
        h2_data_mass_observable = detach(
            h3_data.Project3D("zx")
        )
        h2_data_mass_observable.SetName(
            make_name("data_mass_observable")
        )

    finally:
        # Restore the pT range of the supplied input histogram.
        data_pt_axis.SetRange(
            original_data_pt_first_bin,
            original_data_pt_last_bin,
        )

    # ------------------------------------------------------------------
    # Project the observable in lower sideband, signal region,
    # and upper sideband.
    # ------------------------------------------------------------------
    mass_axis = h3_data.GetXaxis()

    lower_sb_first_bin, lower_sb_last_bin = get_interval_bins(
        mass_axis,
        lower_sideband_low,
        lower_sideband_high,
    )

    signal_first_bin, signal_last_bin = get_interval_bins(
        mass_axis,
        lower_sideband_high,
        upper_sideband_low,
    )

    upper_sb_first_bin, upper_sb_last_bin = get_interval_bins(
        mass_axis,
        upper_sideband_low,
        upper_sideband_high,
    )

    h_observable_sideband_low = detach(
        h2_data_mass_observable.ProjectionY(
            make_name("observable_sideband_low"),
            lower_sb_first_bin,
            lower_sb_last_bin,
        )
    )

    h_observable_signal_region = detach(
        h2_data_mass_observable.ProjectionY(
            make_name("observable_signal_region"),
            signal_first_bin,
            signal_last_bin,
        )
    )

    h_observable_sideband_high = detach(
        h2_data_mass_observable.ProjectionY(
            make_name("observable_sideband_high"),
            upper_sb_first_bin,
            upper_sb_last_bin,
        )
    )

    # ------------------------------------------------------------------
    # Linearly interpolate sideband shapes to the signal mass midpoint.
    # ------------------------------------------------------------------
    midpoint_lower_sideband = 0.5 * (
        lower_sideband_low + lower_sideband_high
    )
    midpoint_signal_region = 0.5 * (
        lower_sideband_high + upper_sideband_low
    )
    midpoint_upper_sideband = 0.5 * (
        upper_sideband_low + upper_sideband_high
    )

    interpolation_parameter = (
        (midpoint_signal_region - midpoint_lower_sideband)
        / (midpoint_upper_sideband - midpoint_lower_sideband)
    )

    h_observable_background = detach(
        h_observable_sideband_low.Clone(
            make_name("observable_background")
        )
    )
    h_observable_background.Scale(1.0 - interpolation_parameter)
    h_observable_background.Add(
        h_observable_sideband_high,
        interpolation_parameter,
    )

    # ------------------------------------------------------------------
    # Project MC observable in the same pT interval.
    # ------------------------------------------------------------------
    mc_pt_axis = h2_mc.GetXaxis()

    mc_pt_first_bin, mc_pt_last_bin = get_interval_bins(
        mc_pt_axis,
        low_pt,
        high_pt,
    )

    h_observable_mc_signal = detach(
        h2_mc.ProjectionY(
            make_name("observable_mc_signal"),
            mc_pt_first_bin,
            mc_pt_last_bin,
        )
    )

    # ------------------------------------------------------------------
    # Enforce identical observable binning before the template is built.
    # ------------------------------------------------------------------
    check_compatible_binning(
        h_observable_background,
        h_observable_mc_signal,
    )

    # ------------------------------------------------------------------
    # Normalize all shapes and construct the final template.
    # ------------------------------------------------------------------
    h_observable_background_normalized = normalized_clone(
        h_observable_background,
        "observable_background_normalized",
    )

    h_observable_mc_signal_normalized = normalized_clone(
        h_observable_mc_signal,
        "observable_mc_signal_normalized",
    )

    h_observable_data_signal_normalized = normalized_clone(
        h_observable_signal_region,
        "observable_data_signal_normalized",
    )

    h_observable_template = detach(
        h_observable_background_normalized.Clone(
            make_name("observable_template")
        )
    )
    h_observable_template.Scale(background_fraction)
    h_observable_template.Add(
        h_observable_mc_signal_normalized,
        signal_fraction,
    )

    h_observable_template.SetTitle(
        f"{observable_label} template;{observable_label};Normalized entries"
    )

    result = {
        "pt_range": (low_pt, high_pt),
        "mass_thresholds": tuple(mass_thresholds),
        "signal_to_background": signal_to_background,
        "signal_fraction": signal_fraction,
        "background_fraction": background_fraction,
        "interpolation_parameter": interpolation_parameter,
        "h2_data_mass_observable": h2_data_mass_observable,
        "h_observable_sideband_low": h_observable_sideband_low,
        "h_observable_sideband_high": h_observable_sideband_high,
        "h_observable_signal_region": h_observable_signal_region,
        "h_observable_background": h_observable_background,
        "h_observable_mc_signal": h_observable_mc_signal,
        "h_observable_background_normalized": (
            h_observable_background_normalized
        ),
        "h_observable_mc_signal_normalized": (
            h_observable_mc_signal_normalized
        ),
        "h_observable_data_signal_normalized": (
            h_observable_data_signal_normalized
        ),
        "h_observable_template": h_observable_template,
        "canvases": {},
        "legends": {},
        "pads": {},
        "lines": {},
    }

    # ------------------------------------------------------------------
    # Diagnostic plots.
    # ------------------------------------------------------------------
    if make_plots:
        palette = create_seaborn_palette()
        color_lower_sideband = palette[4].GetNumber()
        color_upper_sideband = palette[5].GetNumber()
        color_background = palette[2].GetNumber()
        color_mc_signal = palette[3].GetNumber()
        color_template = palette[1].GetNumber()
        color_data_signal = palette[0].GetNumber()

        for histogram, color in [
            (h_observable_sideband_low, color_lower_sideband),
            (h_observable_sideband_high, color_upper_sideband),
            (h_observable_background, color_background),
            (h_observable_background_normalized, color_background),
            (h_observable_mc_signal_normalized, color_mc_signal),
            (h_observable_template, color_template),
            (h_observable_data_signal_normalized, color_data_signal),
            (h_observable_signal_region, color_data_signal),
        ]:
            histogram.SetLineColor(color)
            histogram.SetLineWidth(2)

        # --------------------------------------------------------------
        # Canvas: distributions in the three mass regions.
        # --------------------------------------------------------------
        c_mass_regions = r.TCanvas(
            make_name("canvas_mass_regions"),
            "Observable distributions in mass regions",
            1800,
            550,
        )
        c_mass_regions.Divide(3, 1)

        mass_region_plots = [
            (
                h_observable_signal_region,
                "Data: signal mass region",
            ),
            (
                h_observable_sideband_low,
                "Data: lower-mass sideband",
            ),
            (
                h_observable_sideband_high,
                "Data: upper-mass sideband",
            ),
        ]

        for pad_number, (histogram, title) in enumerate(
            mass_region_plots,
            start=1,
        ):
            c_mass_regions.cd(pad_number)

            histogram.SetTitle(title)
            histogram.GetXaxis().SetTitle(observable_label)
            histogram.GetYaxis().SetTitle("Entries")
            histogram.SetStats(0)
            histogram.Draw("hist")

            if log_y:
                r.gPad.SetLogy()

        c_mass_regions.Update()
        c_mass_regions.Draw()

        # --------------------------------------------------------------
        # Canvas: sideband interpolation.
        # --------------------------------------------------------------
        c_background = r.TCanvas(
            make_name("canvas_background"),
            "Interpolated background",
            850,
            650,
        )
        c_background.cd()

        h_observable_background.SetTitle(
            f"Sideband interpolation;{observable_label};Entries"
        )
        h_observable_sideband_low.SetTitle("Background")
        h_observable_sideband_low.Draw("hist")
        h_observable_sideband_high.Draw("hist same")
        h_observable_background.Draw("hist same")

        legend_background = r.TLegend(0.48, 0.65, 0.88, 0.86)
        legend_background.SetBorderSize(0)
        legend_background.SetFillStyle(0)

        legend_background.AddEntry(
            h_observable_sideband_low,
            "Lower-mass sideband",
            "l",
        )
        legend_background.AddEntry(
            h_observable_sideband_high,
            "Upper-mass sideband",
            "l",
        )
        legend_background.AddEntry(
            h_observable_background,
            f"Interpolated background, alpha = {interpolation_parameter:.3f}",
            "l",
        )
        legend_background.Draw()

        if log_y:
            r.gPad.SetLogy()

        c_background.Update()
        c_background.Draw()

        # --------------------------------------------------------------
        # Canvas: template compared to data in the signal mass region.
        # --------------------------------------------------------------
        c_template = r.TCanvas(
            make_name("canvas_template"),
            "Template comparison with residuals",
            850,
            800,
        )

        template_top_pad = r.TPad(
            make_name("template_top_pad"),
            "Template comparison",
            0.0,
            0.30,
            1.0,
            1.0,
        )

        template_residual_pad = r.TPad(
            make_name("template_residual_pad"),
            "Relative residuals",
            0.0,
            0.0,
            1.0,
            0.30,
        )

        template_top_pad.SetBottomMargin(0.02)
        template_top_pad.SetLeftMargin(0.12)
        template_top_pad.SetRightMargin(0.05)

        template_residual_pad.SetTopMargin(0.04)
        template_residual_pad.SetBottomMargin(0.30)
        template_residual_pad.SetLeftMargin(0.12)
        template_residual_pad.SetRightMargin(0.05)

        template_top_pad.Draw()
        template_residual_pad.Draw()

        # --------------------------------------------------------------
        # Upper pad: normalized template and normalized signal-region data.
        # --------------------------------------------------------------
        template_top_pad.cd()

        h_observable_template.SetTitle("")
        h_observable_template.SetStats(0)
        if x_range_comparison is not None:
            h_observable_template.GetXaxis().SetRangeUser(x_range_comparison[0], x_range_comparison[1])
        h_observable_template.GetXaxis().SetLabelSize(0.0)
        h_observable_template.GetXaxis().SetTitleSize(0.0)
        h_observable_template.GetYaxis().SetTitle("Normalized entries")
        h_observable_template.GetYaxis().SetTitleSize(0.045)
        h_observable_template.GetYaxis().SetLabelSize(0.045)
        h_observable_template.GetYaxis().SetTitleOffset(1.05)

        h_observable_data_signal_normalized.SetStats(0)
        if x_range_comparison is not None:
            h_observable_data_signal_normalized.GetXaxis().SetRangeUser(x_range_comparison[0], x_range_comparison[1])

        h_observable_background_normalized.SetStats(0)
        if x_range_comparison is not None:
            h_observable_background_normalized.GetXaxis().SetRangeUser(x_range_comparison[0], x_range_comparison[1])

        h_observable_template.Draw("hist")
        h_observable_background_normalized.Draw("hist same")
        h_observable_data_signal_normalized.Draw("e1 same")

        legend_template = r.TLegend(0.42, 0.64, 0.88, 0.86)
        legend_template.SetBorderSize(0)
        legend_template.SetFillStyle(0)

        legend_template.AddEntry(
            h_observable_template,
            (
                f"Template: S/B = {signal_to_background:.4g} => "
                f"signal fraction = {signal_fraction:.2%}"
            ),
            "l",
        )
        legend_template.AddEntry(
            h_observable_background_normalized,
            "Interpolated background",
            "l",
        )
        legend_template.AddEntry(
            h_observable_data_signal_normalized,
            "Data in signal mass region (normalized)",
            "lep",
        )
        legend_template.Draw()

        if log_y:
            template_top_pad.SetLogy()

        # --------------------------------------------------------------
        # Lower pad: relative residuals.
        # --------------------------------------------------------------
        template_residual_pad.cd()

        h_observable_relative_residual = detach(
            h_observable_data_signal_normalized.Clone(
                make_name("observable_relative_residual")
            )
        )
        h_observable_relative_residual.Reset("ICES")
        h_observable_relative_residual.SetTitle("")
        h_observable_relative_residual.SetStats(0)
        if x_range_comparison is not None:
            h_observable_relative_residual.GetXaxis().SetRangeUser(x_range_comparison[0], x_range_comparison[1])

        residual_values = []

        for bin_index in range(
            1,
            h_observable_relative_residual.GetNbinsX() + 1,
        ):
            data_value = h_observable_data_signal_normalized.GetBinContent(
                bin_index
            )
            data_error = h_observable_data_signal_normalized.GetBinError(
                bin_index
            )
            template_value = h_observable_template.GetBinContent(
                bin_index
            )

            # A residual is undefined if the template bin is empty.
            if template_value <= 0.0:
                h_observable_relative_residual.SetBinContent(
                    bin_index,
                    0.0,
                )
                h_observable_relative_residual.SetBinError(
                    bin_index,
                    0.0,
                )
                continue

            relative_residual = (
                data_value - template_value
            ) / template_value

            # Data statistical uncertainty only.
            relative_residual_error = data_error / template_value

            h_observable_relative_residual.SetBinContent(
                bin_index,
                relative_residual,
            )
            h_observable_relative_residual.SetBinError(
                bin_index,
                relative_residual_error,
            )

            residual_values.append(
                abs(relative_residual) + relative_residual_error
            )

        h_observable_relative_residual.SetMarkerStyle(20)
        h_observable_relative_residual.SetMarkerSize(0.7)
        h_observable_relative_residual.SetMarkerColor(r.kBlack)
        h_observable_relative_residual.SetLineColor(r.kBlack)

        h_observable_relative_residual.GetXaxis().SetTitle(
            observable_label
        )
        h_observable_relative_residual.GetYaxis().SetTitle(
            "(Data - Template) / Template"
        )

        h_observable_relative_residual.GetXaxis().SetTitleSize(0.10)
        h_observable_relative_residual.GetXaxis().SetLabelSize(0.08)
        h_observable_relative_residual.GetXaxis().SetTitleOffset(1.05)
        h_observable_relative_residual.GetXaxis().SetLabelOffset(0.02)

        h_observable_relative_residual.GetYaxis().SetTitleSize(0.07)
        h_observable_relative_residual.GetYaxis().SetLabelSize(0.08)
        h_observable_relative_residual.GetYaxis().SetTitleOffset(0.4)
        h_observable_relative_residual.GetYaxis().SetLabelOffset(0.02)
        h_observable_relative_residual.GetYaxis().SetNdivisions(505)

        # Choose a symmetric range around zero based on visible residuals.
        residual_limit = max(residual_values, default=0.5)
        residual_limit = max(0.25, 1.15 * residual_limit)

        h_observable_relative_residual.GetYaxis().SetRangeUser(
            -residual_limit,
            residual_limit,
        )
        h_observable_relative_residual.Draw("e1")

        residual_zero_line = r.TLine(
            h_observable_relative_residual.GetXaxis().GetXmin(),
            0.0,
            h_observable_relative_residual.GetXaxis().GetXmax(),
            0.0,
        )
        residual_zero_line.SetLineStyle(2)
        residual_zero_line.SetLineWidth(2)
        residual_zero_line.SetLineColor(r.kGray + 2)
        residual_zero_line.Draw("same")

        c_template.Update()
        c_template.Draw()

        result["h_observable_relative_residual"] = (
            h_observable_relative_residual
        )

        # --------------------------------------------------------------
        # Canvas: individual normalized template components.
        # --------------------------------------------------------------
        c_components = r.TCanvas(
            make_name("canvas_components"),
            "Normalized template components",
            850,
            650,
        )
        c_components.cd()

        h_observable_background_normalized.SetTitle(
            f"Normalized template components;"
            f"{observable_label};Normalized entries"
        )
        h_observable_background_normalized.Draw("hist")
        h_observable_mc_signal_normalized.Draw("hist same")
        h_observable_template.Draw("hist same")

        legend_components = r.TLegend(0.43, 0.62, 0.88, 0.86)
        legend_components.SetBorderSize(0)
        legend_components.SetFillStyle(0)

        legend_components.AddEntry(
            h_observable_background_normalized,
            "Interpolated background shape",
            "l",
        )
        legend_components.AddEntry(
            h_observable_mc_signal_normalized,
            "MC signal shape",
            "l",
        )
        legend_components.AddEntry(
            h_observable_template,
            "Final template",
            "l",
        )
        legend_components.Draw()

        if log_y:
            r.gPad.SetLogy()

        c_components.Update()
        c_components.Draw()

        result["canvases"] = {
            "mass_regions": c_mass_regions,
            "background_interpolation": c_background,
            "template_comparison": c_template,
            "template_components": c_components,
        }

        result["legends"] = {
            "background_interpolation": legend_background,
            "template_comparison": legend_template,
            "template_components": legend_components,
        }

        result["pads"] = {
            "template_comparison_top": template_top_pad,
            "template_comparison_residual": template_residual_pad,
        }

        result["lines"] = {
            "template_residual_zero": residual_zero_line,
        }

    return result

def fit_nonprompt_fraction(
    hData,
    hBkg,
    hPrompt,
    hNonPrompt,
    s_over_b,
    f_B_low=0.0,
    f_B_high=1.0,
    canvas_name="cLxyTemplateFit",
    variable_name="Lxy",
    logy=False,
    title="",
    x_title=0.5,
    show_residuals=False,
):
    """
    One-parameter Lxy template fit without RooFit.

    Parameters
    ----------
    hData : ROOT.TH1
        Data Lxy histogram, containing observed candidate counts.

    hBkg, hPrompt, hNonPrompt : ROOT.TH1
        Background, prompt-MC, and non-prompt-MC templates.
        The function normalizes them internally. They must have the
        same binning as hData and contain no negative bin contents.

    s_over_b : float
        Fixed signal-to-background ratio, S/B, from an external data fit.

    f_B_low : float
        Lower limit for the fitted non-prompt fraction

    f_B_high : float
        Higher limit for the fitted non-prompt fraction

    canvas_name : str
        Name assigned to the ROOT canvas and output histograms.

    logy : bool
        Use a logarithmic y-axis in the plot.

    Returns
    -------
    dict
        Contains the fitted fB, uncertainties, canvas, and fitted
        component TH1 histograms.
    """

    if s_over_b < 0.0:
        raise ValueError("s_over_b must be non-negative.")

    # Convert fixed S/B into the fixed total signal fraction.
    #
    # Fsig = S / (S + B) = (S/B) / (1 + S/B)
    fsig = float(s_over_b) / (1.0 + float(s_over_b))
    fbkg = 1.0 - fsig

    # ------------------------------------------------------------------
    # Check common binning
    # ------------------------------------------------------------------
    nbins = hData.GetNbinsX()
    xmin = hData.GetXaxis().GetXmin()
    xmax = hData.GetXaxis().GetXmax()

    for h in (hBkg, hPrompt, hNonPrompt):
        for i in range(1, nbins + 2):
            edge_data = hData.GetXaxis().GetBinLowEdge(i)
            edge_template = h.GetXaxis().GetBinLowEdge(i)

            if abs(edge_data - edge_template) > 1e-12:
                raise ValueError(
                    "Histogram bin edges differ at edge %d: "
                    "data=%g, template=%g for component %s"
                    % (i, edge_data, edge_template, h.GetName())
                )

    data = [float(hData.GetBinContent(i)) for i in range(1, nbins + 1)]
    bkg = [float(hBkg.GetBinContent(i)) for i in range(1, nbins + 1)]
    prompt = [float(hPrompt.GetBinContent(i)) for i in range(1, nbins + 1)]
    nonprompt = [float(hNonPrompt.GetBinContent(i)) for i in range(1, nbins + 1)]

    if any(x < 0.0 for x in data):
        raise ValueError("Data histogram contains negative bin contents.")

    if any(x < 0.0 for x in bkg + prompt + nonprompt):
        raise ValueError("Template histograms must not contain negative bin contents.")

    nData = sum(data)

    if nData <= 0.0:
        raise ValueError("Data histogram has zero total integral.")

    # Normalize templates to bin probabilities.
    bkg_norm = sum(bkg)
    prompt_norm = sum(prompt)
    nonprompt_norm = sum(nonprompt)

    if bkg_norm <= 0.0:
        raise ValueError("Background template has zero integral.")
    if prompt_norm <= 0.0:
        raise ValueError("Prompt template has zero integral.")
    if nonprompt_norm <= 0.0:
        raise ValueError("Non-prompt template has zero integral.")

    bkg = [x / bkg_norm for x in bkg]
    prompt = [x / prompt_norm for x in prompt]
    nonprompt = [x / nonprompt_norm for x in nonprompt]

    # ------------------------------------------------------------------
    # Binned Poisson negative log likelihood, up to an additive constant.
    # ------------------------------------------------------------------
    def nll(fB):
        result = 0.0

        for observed, b, p, np in zip(data, bkg, prompt, nonprompt):
            probability = (
                fbkg * b
                + fsig * ((1.0 - fB) * p + fB * np)
            )

            expected = nData * probability

            if expected <= 0.0:
                if observed > 0.0:
                    return float("inf")
                continue

            result += expected
            if observed > 0.0:
                result -= observed * math.log(expected)

        return result

    # Fit the sole parameter in its physical range.
    minimization = minimize_scalar(
        nll,
        bounds=(f_B_low, f_B_high),
        method="bounded",
        options={"xatol": 1e-10},
    )

    fB = float(minimization.x)
    nll_min = float(minimization.fun)

    # ------------------------------------------------------------------
    # Compare the best fit against the null hypothesis fB = 0.
    # ------------------------------------------------------------------
    nll_fB0 = nll(0.0)

    q0 = max(0.0, 2.0 * (nll_fB0 - nll_min))

    # Since fB is physically constrained to fB >= 0 and fB=0 is on the
    # boundary, use the asymptotic half-chi-square distribution:
    #
    # p0 = 0.5 * P(chi2 with 1 dof >= q0)
    #
    # Equivalent one-sided Gaussian significance: Z = sqrt(q0).
    if q0 > 0.0:
        p_fB0 = 0.5 * math.erfc(math.sqrt(q0 / 2.0))
        significance = math.sqrt(q0)
    else:
        p_fB0 = 1.0
        significance = 0.0

    # ------------------------------------------------------------------
    # Asymmetric 68% profile-likelihood interval:
    # Delta(-2 ln L) = 1, equivalently Delta(NLL) = 0.5.
    # ------------------------------------------------------------------
    nll_target = nll_min + 0.5

    def profile_root(x):
        return nll(x) - nll_target

    if fB <= 1e-8:
        fB_low = 0.0
    else:
        try:
            fB_low = brentq(profile_root, 0.0, fB)
        except ValueError:
            fB_low = 0.0

    if fB >= 1.0 - 1e-8:
        fB_high = 1.0
    else:
        try:
            fB_high = brentq(profile_root, fB, 1.0)
        except ValueError:
            fB_high = 1.0

    # ------------------------------------------------------------------
    # Make histograms for the fitted components.
    # ------------------------------------------------------------------
    hTotal = hData.Clone(canvas_name + "_total")
    hFitBkg = hData.Clone(canvas_name + "_bkg")
    hFitPrompt = hData.Clone(canvas_name + "_prompt")
    hFitNonPrompt = hData.Clone(canvas_name + "_nonprompt")

    for h in (hTotal, hFitBkg, hFitPrompt, hFitNonPrompt):
        h.Reset("ICES")
        h.SetDirectory(0)

    n_bkg = nData * fbkg
    n_prompt = nData * fsig * (1.0 - fB)
    n_nonprompt = nData * fsig * fB

    for i in range(1, nbins + 1):
        j = i - 1

        y_bkg = n_bkg * bkg[j]
        y_prompt = n_prompt * prompt[j]
        y_nonprompt = n_nonprompt * nonprompt[j]

        hFitBkg.SetBinContent(i, y_bkg)
        hFitPrompt.SetBinContent(i, y_prompt)
        hFitNonPrompt.SetBinContent(i, y_nonprompt)
        hTotal.SetBinContent(i, y_bkg + y_prompt + y_nonprompt)

    # ------------------------------------------------------------------
    # Global goodness of fit: Baker-Cousins likelihood-ratio chi-square.
    #
    # chi2_BC = 2 * sum_i [ mu_i - n_i + n_i*log(n_i/mu_i) ]
    #
    # This is appropriate for the binned Poisson likelihood used above.
    # ------------------------------------------------------------------
    chi2_bc = 0.0
    n_bins_used = 0

    for i in range(1, nbins + 1):
        observed = hData.GetBinContent(i)
        expected = hTotal.GetBinContent(i)

        # Ignore bins that have neither observed nor predicted entries.
        if observed <= 0.0 and expected <= 0.0:
            continue

        n_bins_used += 1

        if expected <= 0.0:
            # This should not occur for a viable fit if observed > 0.
            chi2_bc = float("inf")
            break

        if observed > 0.0:
            chi2_bc += 2.0 * (
                expected
                - observed
                + observed * math.log(observed / expected)
            )
        else:
            chi2_bc += 2.0 * expected

    # One free fitted parameter: fB.
    #
    # S/B is externally fixed, as is the overall histogram normalization.
    ndf = n_bins_used - 1

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    palette = create_seaborn_palette()

    hData.SetMarkerStyle(20)
    hData.SetStats(0)
    hData.SetMarkerSize(0.9)
    hData.SetLineColor(r.kBlack)

    hTotal.SetLineColor(palette[0].GetNumber())
    hTotal.SetStats(0)
    hTotal.SetLineWidth(3)

    hFitBkg.SetLineColor(palette[1].GetNumber())
    hFitBkg.SetStats(0)
    hFitBkg.SetLineStyle(2)
    hFitBkg.SetLineWidth(3)

    hFitPrompt.SetLineColor(palette[2].GetNumber())
    hFitPrompt.SetStats(0)
    hFitPrompt.SetLineStyle(2)
    hFitPrompt.SetLineWidth(3)

    hFitNonPrompt.SetLineColor(palette[3].GetNumber())
    hFitNonPrompt.SetStats(0)
    hFitNonPrompt.SetLineStyle(2)
    hFitNonPrompt.SetLineWidth(3)

    # ------------------------------------------------------------------
    # Residuals / pulls:
    #
    # pull_i = (N_data,i - N_fit,i) / sqrt(N_fit,i)
    # ------------------------------------------------------------------
    hResidual = hData.Clone(canvas_name + "_residuals")
    hResidual.Reset("ICES")
    hResidual.SetDirectory(0)

    for i in range(1, nbins + 1):
        observed = hData.GetBinContent(i)
        expected = hTotal.GetBinContent(i)

        if expected > 0.0:
            residual = (observed - expected) / math.sqrt(expected)
        else:
            residual = 0.0

        hResidual.SetBinContent(i, residual)
        hResidual.SetBinError(i, 0.0)

    hResidual.SetMarkerStyle(20)
    hResidual.SetMarkerSize(0.8)
    hResidual.SetLineColor(r.kBlack)

    # ------------------------------------------------------------------
    # Canvas with upper fit pad and lower residual pad
    # ------------------------------------------------------------------
    canvas = r.TCanvas(f"{canvas_name}_{uuid.uuid4().hex[:10]}", canvas_name, 900, 800)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.03)

    if show_residuals:
        padTop = r.TPad(
            canvas_name + "_top",
            canvas_name + "_top",
            0.0, 0.30, 1.0, 1.0
        )

        padBottom = r.TPad(
            canvas_name + "_bottom",
            canvas_name + "_bottom",
            0.0, 0.0, 1.0, 0.30
        )

        padTop.SetLeftMargin(0.12)
        padTop.SetRightMargin(0.05)
        padTop.SetTopMargin(0.07)
        padTop.SetBottomMargin(0.02)

        padBottom.SetLeftMargin(0.12)
        padBottom.SetRightMargin(0.05)
        padBottom.SetTopMargin(0.03)
        padBottom.SetBottomMargin(0.32)

        if logy:
            padTop.SetLogy()

        padTop.Draw()
        padBottom.Draw()
    elif logy:
        r.gPad.SetLogy() 

    # ------------------------------------------------------------------
    # Top pad: data and templates
    # ------------------------------------------------------------------
    # Clones of histograms to show density
    hDataPlot = hData.Clone("hDataPlot")
    hDataPlot.Scale(1.0, "width")
    hTotalPlot = hTotal.Clone("hTotalPlot")
    hTotalPlot.Scale(1.0, "width")
    hFitBkgPlot = hFitBkg.Clone("hFitBkgPlot")
    hFitBkgPlot.Scale(1.0, "width")
    hFitPromptPlot = hFitPrompt.Clone("hFitPromptPlot")
    hFitPromptPlot.Scale(1.0, "width")
    hFitNonPromptPlot = hFitNonPrompt.Clone("hFitNonPromptPlot")
    hFitNonPromptPlot.Scale(1.0, "width")

    if show_residuals:
        padTop.cd()

    ymax = max(hDataPlot.GetMaximum(), hTotalPlot.GetMaximum())
    hDataPlot.SetMaximum(1.35 * ymax)
    hDataPlot.SetMinimum(1 if logy else 0.0)

    hDataPlot.SetTitle("")

    hDataPlot.GetYaxis().SetTitle(f"Candidate density")
    hDataPlot.GetYaxis().SetTitleSize(0.045)
    hDataPlot.GetYaxis().SetLabelSize(0.04)
    hDataPlot.GetXaxis().SetTitleSize(0.045)
    hDataPlot.GetXaxis().SetLabelSize(0.04)

    # Hide the x-axis labels in the upper pad.
    if show_residuals:
        hDataPlot.GetXaxis().SetLabelSize(0.0)
        hDataPlot.GetXaxis().SetTitleSize(0.0)

    hDataPlot.Draw("E1")
    hTotalPlot.Draw("HIST SAME")
    hFitBkgPlot.Draw("HIST SAME")
    hFitPromptPlot.Draw("HIST SAME")
    hFitNonPromptPlot.Draw("HIST SAME")
    hDataPlot.Draw("E1 SAME")

    # legend = r.TLegend(0.15, 0.60, 0.6, 0.88)
    legend = r.TLegend(0.15, 0.12, 0.6, 0.40)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(hDataPlot, "Data", "lep")
    legend.AddEntry(hTotalPlot, "Total fit", "l")
    legend.AddEntry(hFitBkgPlot, "Background", "l")
    legend.AddEntry(hFitPromptPlot, "Prompt D^{0}", "l")
    legend.AddEntry(hFitNonPromptPlot, "Non-prompt D^{0}", "l")
    legend.Draw()

    label = r.TLatex()
    label.SetNDC(True)
    label.SetTextSize(0.036)
    label.SetTextFont(42)

    x_text = 0.64
    y_text = 0.83
    y_text_delta = 0.06 
    label.DrawLatex(
        x_title,
        y_text,
        "#bf{%s}" % title
    )
    label.DrawLatex(
        x_text,
        y_text - y_text_delta,
        "#scale[1.0]{f_{B} = %.3f^{+%.3f}_{-%.3f}}" % (
            fB,
            fB_high - fB,
            fB - fB_low,
        ),
    )
    label.DrawLatex(
        x_text, 
        y_text - 2*y_text_delta, 
        "#scale[1.0]{S/B = %.3f (fixed)}" % s_over_b,
    )
    label.DrawLatex(
        x_text,
        y_text - 3*y_text_delta,
        #"#chi^{2}_{BC}/ndf = %.1f / %d" % (chi2_bc, ndf),
        "#scale[1.0]{#chi^{2}/ndf = %.2f}" % (chi2_bc/ndf),
    )
    """
    label.DrawLatex(
        x_text,
        0.71,
        "-2#Delta ln L(f_{B}=0) = %.2f" % q0,
    )

    label.DrawLatex(
        x_text,
        0.66,
        "Preference for f_{B}>0: %.2f #sigma" % significance,
    )
    """

    # ------------------------------------------------------------------
    # Bottom pad: standardized residuals
    # ------------------------------------------------------------------
    if show_residuals:
        padBottom.cd()

        max_abs_residual = max(
            abs(hResidual.GetBinContent(i))
            for i in range(1, nbins + 1)
        )

        # Always show at least the range [-3, 3].
        residual_range = max(3.0, 1.20 * max_abs_residual)

        hResidual.SetMinimum(-residual_range)
        hResidual.SetMaximum(+residual_range)

        hResidual.GetYaxis().SetTitle("(Data-Fit)/#sqrt{Fit}")
        hResidual.GetYaxis().SetNdivisions(505)
        hResidual.GetYaxis().SetTitleSize(0.10)
        hResidual.GetYaxis().SetTitleOffset(0.52)
        hResidual.GetYaxis().SetLabelSize(0.085)

        hResidual.GetXaxis().SetTitle(hData.GetXaxis().GetTitle())
        hResidual.GetXaxis().SetTitleSize(0.12)
        hResidual.GetXaxis().SetTitleOffset(1.05)
        hResidual.GetXaxis().SetLabelSize(0.10)

        hResidual.Draw("P")

        zeroLine = r.TLine(xmin, 0.0, xmax, 0.0)
        zeroLine.SetLineColor(r.kGray + 2)
        zeroLine.SetLineWidth(2)
        zeroLine.Draw("SAME")

        plusOneLine = r.TLine(xmin, 1.0, xmax, 1.0)
        minusOneLine = r.TLine(xmin, -1.0, xmax, -1.0)

        for line in (plusOneLine, minusOneLine):
            line.SetLineColor(r.kGray + 2)
            line.SetLineStyle(2)
            line.Draw("SAME")

    canvas.cd()
    canvas.Modified()
    canvas.Update()

    result = {
        "fB": fB,
        "fB_errlo": fB - fB_low,
        "fB_errhi": fB_high - fB,
        "nll": nll_min,
        "s_over_b": s_over_b,
        "signal_fraction": fsig,
        "canvas": canvas,
        "data": hData,
        "total": hTotal,
        "bkg": hFitBkg,
        "prompt": hFitPrompt,
        "nonprompt": hFitNonPrompt,
        "legend": legend,
    }
    if show_residuals:
        result.update({
            "residuals": hResidual,
            "pad_top": padTop,
            "pad_bottom": padBottom,
            "lines": (zeroLine, plusOneLine, minusOneLine)}
        )
    return result

def crop_hist_x(h, xcut, name=None):
    """
    Return a new TH1 histogram containing bins up to xcut.

    Only complete bins are retained: a bin is included if its upper edge
    is <= xcut. Underflow and overflow contents/errors are copied too.

    Parameters
    ----------
    h : ROOT.TH1
        Input histogram.
    xcut : float
        Upper x boundary.
    name : str, optional
        Name for the output histogram.

    Returns
    -------
    ROOT.TH1D
        Cropped histogram.
    """
    if name is None:
        name = h.GetName() + "_cropped"

    xaxis = h.GetXaxis()
    nbins = h.GetNbinsX()

    # Find final fully included bin.
    last_bin = 0
    for ibin in range(1, nbins + 1):
        if xaxis.GetBinUpEdge(ibin) <= xcut:
            last_bin = ibin
        else:
            break

    if last_bin == 0:
        raise ValueError(
            f"xcut={xcut} is below the upper edge of the first bin "
            f"({xaxis.GetBinUpEdge(1)})"
        )

    # Preserve the original bin edges, including variable-width bins.
    edges = [xaxis.GetBinLowEdge(1)]
    for ibin in range(1, last_bin + 1):
        edges.append(xaxis.GetBinUpEdge(ibin))

    out = r.TH1D(
        name,
        h.GetTitle(),
        last_bin,
        array("d", edges),
    )
    out.SetDirectory(0)

    # Preserve axis titles.
    out.GetXaxis().SetTitle(xaxis.GetTitle())
    out.GetYaxis().SetTitle(h.GetYaxis().GetTitle())

    # Preserve bin content and uncertainties.
    out.Sumw2()
    for ibin in range(0, last_bin + 2):
        out.SetBinContent(ibin, h.GetBinContent(ibin))
        out.SetBinError(ibin, h.GetBinError(ibin))

    # Preserve common display settings.
    out.SetLineColor(h.GetLineColor())
    out.SetLineStyle(h.GetLineStyle())
    out.SetLineWidth(h.GetLineWidth())
    out.SetMarkerColor(h.GetMarkerColor())
    out.SetMarkerStyle(h.GetMarkerStyle())
    out.SetMarkerSize(h.GetMarkerSize())
    out.SetFillColor(h.GetFillColor())
    out.SetFillStyle(h.GetFillStyle())

    return out
