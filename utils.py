import ROOT as r
import numpy as np
from array import array
from scipy.stats import poisson

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
