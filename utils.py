import ROOT as r
import numpy as np
from array import array
from scipy.stats import poisson
import matplotlib.pyplot as plt
from boost_histogram import axis
import hist

defaultITSROFlength = 14821e-9 # seconds

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
         150, 1.0, 2.5,
         240, 0.0, 12.0),
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
    """Create a clone of h_ref's binning and fill it from h_src
       by matching (x,y) bin centers.
    """
    xaxis_ref = h_ref.GetXaxis()
    yaxis_ref = h_ref.GetYaxis()

    nx = xaxis_ref.GetNbins()
    ny = yaxis_ref.GetNbins()

    # copy bin edges from reference
    x_edges = [xaxis_ref.GetBinLowEdge(i) for i in range(1, nx+2)]
    y_edges = [yaxis_ref.GetBinLowEdge(i) for i in range(1, ny+2)]

    h_new = r.TH2F(
        h_src.GetName() + name_suffix,
        h_src.GetTitle() + name_suffix,
        nx, array('d', x_edges),
        ny, array('d', y_edges),
    )
    h_new.Sumw2()

    # fill by copying content to matching bins (via centers)
    for ix in range(1, nx+1):
        x = xaxis_ref.GetBinCenter(ix)
        for iy in range(1, ny+1):
            y = yaxis_ref.GetBinCenter(iy)

            # find corresponding bin in the source
            ix_src = h_src.GetXaxis().FindBin(x)
            iy_src = h_src.GetYaxis().FindBin(y)

            c = h_src.GetBinContent(ix_src, iy_src)
            e = h_src.GetBinError(ix_src, iy_src)

            h_new.SetBinContent(ix, iy, c)
            h_new.SetBinError(ix, iy, e)

    return h_new

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
    print(f"Processing {len(run_list)} runs...")
    lumi_file = r.TFile.Open("/media/sigurd/T7/analysis/data/LHC23_PbPb_pass5_train590144/mergedAnalysisResults_good.root")
    lumi_hist_znc = lumi_file.Get('eventselection-run3').Get('luminosity').Get('hLumiZNCafterBCcuts')
    hist = r.TH1D('lumi_ir', 'lumi_ir', len(ir_bin_edges)-1, np.asarray(ir_bin_edges, 'd'))
    hist.SetDirectory(0)
    hist.GetXaxis().SetTitle('ZNC hadronic interaction rate (Hz)')
    hist.GetYaxis().SetTitle('ZNC luminosity (1/µb)')
    for i, (ir_0, ir_1) in enumerate(zip(ir_bin_edges[:-1], ir_bin_edges[1:])):
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

        print(f"  {n_contributing_runs} runs contributed in total {L:.3f} 1/µb of luminosity between {ir_0} kHz and {ir_1} kHz")
        hist.SetBinContent(i+1, L)
    return hist

def plot_run_by_run_ir_lumi(run_list, ir_0, ir_1, lumi_path="/media/sigurd/T7/analysis/data/LHC23_PbPb_pass5_train590144/mergedAnalysisResults_good.root", do_print=False, y_axis='lumi', ncols=5, scale=1.0):
    """
    For every run in run_list, illustrate the contribution to the luminosity between interaction rate ir_0 to ir_1
    """
    lumi_file = r.TFile.Open("/media/sigurd/T7/analysis/data/LHC23_PbPb_pass5_train590144/mergedAnalysisResults_good.root")
    lumi_hist_znc = lumi_file.Get('eventselection-run3').Get('luminosity').Get('hLumiZNCafterBCcuts')

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

    plt.show()

def root_to_hist(hroot):
    nbins = hroot.GetNbinsX()
    edges = np.array([hroot.GetBinLowEdge(i) for i in range(1, nbins + 2)])
    counts = np.array([hroot.GetBinContent(i) for i in range(1, nbins + 1)])
    errors = np.array([hroot.GetBinError(i) for i in range(1, nbins + 1)])

    h = hist.Hist(
        hist.axis.Variable(edges, name=hroot.GetName(), label=hroot.GetTitle()),
        storage=hist.storage.Weight()
    )

    view = h.view()
    view.value[...] = counts
    view.variance[...] = errors**2

    return h

def equal_stat_y_slices(h2, n_slices, start_bin=1):
    """
    h2: TH2 (e.g. TH2F)
    n_slices: desired number of Y slices with equal total entries (over X)

    Returns:
        A list of (ybin_lo, ybin_hi) tuples (1-based, inclusive),
        not using underflow/overflow bins.
    """
    nbins_x = h2.GetNbinsX()
    nbins_y = h2.GetNbinsY()

    # 1) Sum over X for each Y bin (start_bin..nbins_y)
    per_y = []
    total = 0.0
    for j in range(start_bin, nbins_y + 1):
        s = 0.0
        for i in range(1, nbins_x + 1):
            s += h2.GetBinContent(i, j)
        per_y.append(s)
        total += s

    if total <= 0:
        raise RuntimeError("Histogram is empty (no entries in in-range Y bins).")

    # 2) Cumulative along Y
    cumulative = []
    run = 0.0
    for s in per_y:
        run += s
        cumulative.append(run)

    target_per_slice = total / float(n_slices)

    # 3) Find upper-bin edge for each slice (except the last)
    upper_edges = []  # list of Y-bin indices (start_bin..nbins_y) as upper edges
    k = 1  # we’re looking for k * target_per_slice
    for idx in range(len(per_y)):
        j = start_bin + idx
        if k >= n_slices:
            break
        if cumulative[idx] >= k * target_per_slice:
            upper_edges.append(j)
            k += 1

    # Make sure we have n_slices-1 boundaries, then add last at nbins_y
    # If some were missed (e.g. lots of empty bins), just don’t add extras;
    # last slice will absorb remaining bins.
    upper_edges = upper_edges[:n_slices-1]
    upper_edges.append(nbins_y)

    # 4) Convert upper edges into (lo, hi) bin ranges
    slices = []
    prev_hi = start_bin - 1
    for hi in upper_edges:
        lo = prev_hi + 1
        slices.append((lo, hi))
        prev_hi = hi

    return slices
