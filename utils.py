import ROOT as r
import numpy as np

def propagate_error_product(A, B, errA, errB, covAB):
    # Error propagation for the product of two variables r = A*B
    var = (B*errA)**2 + (A*errB)**2 + 2*A*B*covAB
    return np.sqrt(var)

def flip_th2_axes(h):
    hf = r.TH2F(f'{h.GetName()}_flip', f'{h.GetName()}_flip', h.GetNbinsY(), h.GetYaxis().GetXmin(), h.GetYaxis().GetXmax(), h.GetNbinsX(), h.GetXaxis().GetXmin(), h.GetXaxis().GetXmax())
    hf.GetXaxis().SetTitle(h.GetYaxis().GetTitle())
    hf.GetYaxis().SetTitle(h.GetXaxis().GetTitle())
    for i in range(h.GetNbinsX()):
        for j in range(h.GetNbinsY()):
            hf.SetBinContent(j, i, h.GetBinContent(i, j))
    return hf
