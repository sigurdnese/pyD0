import ROOT as r
import numpy as np
from array import array

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
