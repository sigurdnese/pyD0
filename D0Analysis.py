import ROOT as r
import numpy as np

class EfficiencyFactor():
    def __init__(self, numeratorTitle, denominatorTitle, histNumerator, histDenominator):
        self.histogram = histNumerator.Clone()
        self.histogram.Divide(histNumerator, histDenominator, 1, 1, "B")
        self.histogram.SetName(f"eff_{histNumerator.GetName()}_{histDenominator.GetName()}")
        self.histogram.SetTitle(f"eff_{histNumerator.GetName()}_{histDenominator.GetName()}")
        self.titleRaw = "#frac{%s}{%s}" % (numeratorTitle, denominatorTitle)
        self.shortTitle = ""

    def draw_title(self, x, y, size=0.04, includeshort=False):
        self.title = r.TLatex()
        self.title.SetTextSize(size)
        self.title.SetTextAlign(22)
        if includeshort:
            self.title.DrawLatexNDC(x, y, self.titleRaw+"    "+self.shortTitle)
        else:
            self.title.DrawLatexNDC(x, y, self.titleRaw)

    def replace_displayed_title(self, includeshort=False):
        # Replace the displayed title of the histogram with the TeX formatted title
        self.histogram.SetTitle("")
        self.draw_title(0.5, 0.95, size=0.035, includeshort=includeshort)

class FactorizedEfficiency():
    def __init__(self, nFactors):
        if nFactors == 0:
            raise Exception("Need at least one factor!")
        self.nFactors = nFactors
        self.factors = []

    def add_factor(self, numeratorTitle, denominatorTitle, histNumerator, histDenominator):
        self.factors.append(EfficiencyFactor(numeratorTitle, denominatorTitle, histNumerator, histDenominator))
        self.factors[-1].shortTitle = f"({len(self.factors)})"

    def calculate_total_efficiency(self):
        if len(self.factors) != self.nFactors:
            raise Exception(f"Cannot calculate total efficiency: Only have {len(self.factors)} out of {self.nFactors} factors!")
        self.totalEfficiency = self.factors[0].histogram.Clone()
        self.totalEfficiency.SetName(f"totalEfficiency{self.nFactors}Factors")
        self.totalEfficiency.SetTitle(f"Total efficiency calculated from {self.nFactors} factors")
        self.totalEfficiencyTitleRaw = self.factors[0].titleRaw
        self.totalEfficiencyShortTitle = self.factors[0].shortTitle
        for factor in self.factors[1:]:
            self.totalEfficiency.Multiply(self.totalEfficiency, factor.histogram, 1, 1, "B")
            self.totalEfficiencyTitleRaw += "#times%s" % (factor.titleRaw)
            self.totalEfficiencyShortTitle += "#times%s" % (factor.shortTitle)

    def draw_title(self, x, y, size=0.04):
        self.totalEfficiencyTitle = r.TLatex()
        self.totalEfficiencyTitle.SetTextSize(size)
        self.totalEfficiencyTitle.SetTextAlign(22)
        self.totalEfficiencyTitle.DrawLatexNDC(x, y, self.totalEfficiencyTitleRaw)

    def replace_displayed_title(self, size=None, short=False):
        # Replace the displayed title of the histogram with the TeX formatted title
        self.totalEfficiency.SetTitle("")
        if short:
            self.totalEfficiencyTitle = r.TLatex()
            self.totalEfficiencyTitle.SetTextSize(0.04)
            self.totalEfficiencyTitle.SetTextAlign(22)
            self.totalEfficiencyTitle.DrawLatexNDC(0.5, 0.95, self.totalEfficiencyShortTitle)
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

class PtBin():
    """A single pT bin of an analysis"""

    def __init__(self, index, lowerPt, upperPt):
        self.index = index
        self.lowerPt = lowerPt
        self.upperPt = upperPt

    def create_fit_results(self, lowMass, highMass):
        try:
            self.fitChi2Ndf = self.fitFunc.GetChisquare() / self.fitFunc.GetNDF()
        except:
            self.fitChi2Ndf = 0
        try:
            self.reflChi2Ndf = self.reflFunc.GetChisquare() / self.reflFunc.GetNDF()
        except:
            self.reflChi2Ndf = 0
        self.fitMu = self.fitFunc.GetParameter('Mean')
        self.fitSigma = self.fitFunc.GetParameter("Sigma")
        self.nSignal = self.signalFunc.Integral(self.fitMu - 3*self.fitSigma, self.fitMu + 3*self.fitSigma) / self.massPtSlice.GetBinWidth(1)
        print(f"bin {self.index} nSignal = {self.nSignal} was calculated by integration from {self.fitMu - 3*self.fitSigma} to {self.fitMu + 3*self.fitSigma}")
        self.nCombBackground = self.backgroundFunc.Integral(self.fitMu - 3*self.fitSigma, self.fitMu + 3*self.fitSigma) / self.massPtSlice.GetBinWidth(1)
        print(f"bin {self.index} nCombBackground = {self.nCombBackground} was calculated by integration from {self.fitMu - 3*self.fitSigma} to {self.fitMu + 3*self.fitSigma}")
        self.nReflBackground = self.dataReflFunc.Integral(lowMass, highMass) / self.massPtSlice.GetBinWidth(1)
        if (self.dataReflFunc.Eval(lowMass) > 1e-6):
            print(f"WARNING: dataReflFunc({lowMass}) = {self.dataReflFunc.Eval(lowMass)}, normalization of reflected background may be inaccurate")
        if (self.dataReflFunc.Eval(highMass) > 1e-6):
            print(f"WARNING: dataReflFunc({highMass}) = {self.dataReflFunc.Eval(highMass)}, normalization of reflected background may be inaccurate")
        print(f"bin {self.index} nReflBackground = {self.nReflBackground} was calculated by integration from {lowMass} to {highMass}")
        self.nBackground = self.nCombBackground + self.nReflBackground
        self.relativeStatError = np.sqrt(self.nSignal + self.nBackground) / self.nSignal
        
class Analysis():
    """Class containing everything needed to calculate a D0 cross section from O2Physics output"""

    def __init__(self, pathTableReader, pathTableMaker, pathRec, pathGen, pathReflected = None, ptBins = [0., 0.75, 1.5, 2.25, 3., 4., 6., 8., 12.], IsITSUPCMode = 2, old=False):
        self.old = old
        # Files containing the necessary histograms
        self.fileTableReader = r.TFile.Open(pathTableReader)
        self.fileTableMaker = r.TFile.Open(pathTableMaker)
        self.fileRec = r.TFile.Open(pathRec)
        self.fileGen = r.TFile.Open(pathGen)
        if pathReflected is None:
            self.fileReflected = self.fileRec
        else:
            self.fileReflected = r.TFile.Open(pathReflected)
        # pT bins to be used in the differential cross section
        self.ptBinsArray = ptBins
        self.ptBins = []
        for i in range(len(self.ptBinsArray) - 1):
            self.ptBins.append(PtBin(i, self.ptBinsArray[i], self.ptBinsArray[i+1]))
        # Range of rapidity bin to be used
        self.minY = -0.9
        self.maxY = 0.9
        # Get the mass histograms needed for the signal extraction
        self.isITSUPCMode = IsITSUPCMode
        self.prepare_histograms()
        # Set the range of the mass axis for plotting and integrating reflected background
        self.massRange = [0., 4.]
        # Get the integrated luminosity from the bc-selection-task
        self.histLumi = self.fileTableMaker.Get("bc-selection-task").Get("hLumiTCEafterBCcuts")
        self.lumi = self.histLumi.Integral() # 1/µb
        print(f"Integrated luminosity (TCE trigger, after BC cuts): {self.lumi} 1/µb")

    def prepare_histograms(self):
        self.groupNameD0PtMatched = "PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepTauxyzprojCut_KPiFromD0"
        self.groupNameD0Generated = "MCTruthGenAfterBcCuts_D0FS"
        if self.isITSUPCMode == 2:
            self.histD0MassPt = self.fileTableReader.Get("analysis-asymmetric-pairing/output").FindObject("PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepTauxyzprojCut").FindObject("MyMassPtHisto")
        else:
            print(f"Getting histograms from multidim histograms w/ variable self.isITSUPCMode = {self.isITSUPCMode}")
            histD0MassPtIsITSUPCMode = self.fileTableReader.Get("analysis-asymmetric-pairing/output").FindObject("PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepTauxyzprojCut").FindObject("MyMassPtIsITSUPCModeHisto")
            histD0MassPtIsITSUPCMode.GetZaxis().SetRange(self.isITSUPCMode+1, self.isITSUPCMode+1)
            self.histD0MassPt = histD0MassPtIsITSUPCMode.Project3D("yx")
            self.histD0MassPt.SetName(f"MyMassPtHisto_IsITSUPCMode={self.isITSUPCMode}")
            """
            ONLY DATA SHOULD HAVE THE IsITSUPCMode CUT APPLIED, SINCE MC SHOULD NOT BE USED TO CALCULATE THE EFFICIENCY FOR THAT SELECTION
            histD0PtIsITSUPCModeMatched = self.fileRec.Get("analysis-asymmetric-pairing/output").FindObject(self.groupNameD0PtMatched).FindObject("MyPtIsITSUPCModeHisto") 
            self.histD0PtMatched = histD0PtIsITSUPCModeMatched.ProjectionX(f"histD0PtMatched_IsITSUPCMode={self.isITSUPCMode}", self.isITSUPCMode+1, self.isITSUPCMode+1)
            histD0PtYIsITSUPCModeGenerated = self.fileGen.Get("analysis-asymmetric-pairing/output").FindObject(self.groupNameD0Generated).FindObject("MyMcPtYIsITSUPCModeHisto")
            histD0PtYIsITSUPCModeGenerated.GetZaxis().SetRange(self.isITSUPCMode+1, self.isITSUPCMode+1)
            self.histD0PtYGenerated = histD0PtYIsITSUPCModeGenerated.Project3D("yx")
            self.histD0PtYGenerated.SetName(f"MyMcPtYHisto_IsITSUPCMode={self.isITSUPCMode}")
            histD0MassPtIsITSUPCModeReflected = self.fileReflected.Get("analysis-asymmetric-pairing/output").FindObject("PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepTauxyzprojCut_KPiFromD0Reflected").FindObject("MyMassPtIsITSUPCModeHisto")
            histD0MassPtIsITSUPCModeReflected.GetZaxis().SetRange(self.isITSUPCMode+1, self.isITSUPCMode+1)
            self.histD0MassPtReflected = histD0MassPtIsITSUPCModeReflected.Project3D("yx")
            self.histD0MassPtReflected.SetName(f"MyMcReflectedMassPtHisto_IsITSUPCMode={self.isITSUPCMode}")
            """
        self.histD0PtMatched = self.fileRec.Get("analysis-asymmetric-pairing/output").FindObject(self.groupNameD0PtMatched).FindObject("Pt")
        self.histD0PtMatched.SetName("histD0PtMatched")
        self.histD0PtMatched.SetTitle("Reconstructed, matched D0")
        self.histD0PtYGenerated = self.fileGen.Get("analysis-asymmetric-pairing/output").FindObject(self.groupNameD0Generated).FindObject("MyMcPtYHisto")
        self.histD0MassPtReflected = self.fileReflected.Get("analysis-asymmetric-pairing/output").FindObject("PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepTauxyzprojCut_KPiFromD0Reflected").FindObject("MyMassPtHisto")
            
        # Project out our dy bin from the gen histogram
        lowerYBin = self.histD0PtYGenerated.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenerated.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenerated = self.histD0PtYGenerated.ProjectionX(f"projPtMcGen_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenerated.SetTitle(f"Generated D0 after BC cuts in {self.minY} < y < {self.maxY}")

        for i, bin in enumerate(self.ptBins):
            # Create an array of slices of the mass vs pT histogram and the MC reflected mass vs pT histogram
            lowerBin = self.histD0MassPt.GetYaxis().FindBin(bin.lowerPt)
            upperBin = self.histD0MassPt.GetYaxis().FindBin(bin.upperPt) - 1
            hProjectionMass = self.histD0MassPt.ProjectionX(f"projMass_{bin.lowerPt}_{bin.upperPt}", firstybin=lowerBin, lastybin=upperBin)
            hProjectionMass.Rebin(5)
            hProjectionMass.SetTitle(f"Kpi invariant mass, {bin.lowerPt} <= pT < {bin.upperPt} GeV/c")
            bin.massPtSlice = hProjectionMass

            hProjectionMassReflected = self.histD0MassPtReflected.ProjectionX(f"projMassMcReflected_{bin.lowerPt}_{bin.upperPt}", firstybin=lowerBin, lastybin=upperBin)
            hProjectionMassReflected.Rebin(2)
            hProjectionMassReflected.SetTitle(f"MC matched reflected Kpi invariant mass, {bin.lowerPt} <= pT < {bin.upperPt} GeV/c")
            bin.massPtSliceReflected = hProjectionMassReflected

            # Count the number of matched D0 in this bin's pT range
            matchedLowerBin = self.histD0PtMatched.GetXaxis().FindBin(bin.lowerPt)
            matchedUpperBin = self.histD0PtMatched.GetXaxis().FindBin(bin.upperPt) - 1
            bin.nMatched = self.histD0PtMatched.Integral(matchedLowerBin, matchedUpperBin)

    def fit_inv_mass(self, bin, backgroundName, fitRange = [1.64, 2.08], initialParams = []):
        self.ptBins[bin].fitRange = fitRange
        print(f"====== Fitting bin {bin} ({self.ptBins[bin].lowerPt} < pT < {self.ptBins[bin].upperPt} GeV/c) ======")
        # Fit the reflected MC histogram with a double Gaussian
        reflFunc = r.TF1(f"fDoubleGauss_bin{bin}", "[0]*exp(-0.5*((x-[1])/[2])^2) + [3]*exp(-0.5*((x-[4])/[5])^2)", self.massRange[0], self.massRange[1])
        reflFunc.SetParameters(1, self.ptBins[bin].massPtSliceReflected.GetMean(), self.ptBins[bin].massPtSliceReflected.GetRMS()/2, 1, self.ptBins[bin].massPtSliceReflected.GetMean(), self.ptBins[bin].massPtSliceReflected.GetRMS()/2)
        reflFunc.SetParNames("Amp1", "Mean1", "Sigma1", "Amp2", "Mean2", "Sigma2")
        # Require the amplitudes to be positive
        reflFunc.SetParLimits(0, 0, 1e6)
        reflFunc.SetParLimits(3, 0, 1e6)
        # Require reasonable peak positions and widths for the Gaussians
        reflFunc.SetParLimits(1, self.ptBins[bin].massPtSliceReflected.GetXaxis().GetXmin(), self.ptBins[bin].massPtSliceReflected.GetXaxis().GetXmax())
        reflFunc.SetParLimits(4, self.ptBins[bin].massPtSliceReflected.GetXaxis().GetXmin(), self.ptBins[bin].massPtSliceReflected.GetXaxis().GetXmax())
        reflFunc.SetParLimits(2, 0, 0.2)
        reflFunc.SetParLimits(5, 0, 0.2)
        print("---- Fitting MC matched, reflected ----")
        self.ptBins[bin].massPtSliceReflected.Fit(reflFunc, "LR0")
        self.ptBins[bin].reflFunc = reflFunc

        self.ptBins[bin].reflectedRatio = self.ptBins[bin].massPtSliceReflected.GetEntries() / self.ptBins[bin].nMatched

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
        else:
            raise Exception(f"Invalid background function '{backgroundName}'")

        # Fit the histogram
        print("---- Fitting data ----")
        self.ptBins[bin].massPtSlice.Fit(fitFunc, "LR0")

        # Obtain the signal and background functions separately
        signalFunc = r.TF1(f"signalFunc_bin{bin}", "[0]*exp(-0.5*((x-[2])/[1])^2)", self.massRange[0], self.massRange[1])
        signalFunc.SetParameters(fitFunc.GetParameter("Amp"), fitFunc.GetParameter("Sigma"), fitFunc.GetParameter("Mean"))

        if backgroundName == "pol2":
            backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", "([0]*x*x + [1]*x + [2])", self.massRange[0], self.massRange[1])
            backgroundFunc.SetParameters(fitFunc.GetParameter("a"), fitFunc.GetParameter("b"), fitFunc.GetParameter("c"))
        elif backgroundName == "exp":
            backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", "([0]*exp([1]*x))", self.massRange[0], self.massRange[1])
            backgroundFunc.SetParameters(fitFunc.GetParameter("A"), fitFunc.GetParameter("B"))
        else:
            raise Exception(f"Background function '{backgroundName}' not implemented when extracting separate background shape!")
        backgroundFunc.SetLineColor(r.kBlue)

        dataReflFunc = r.TF1(f"dataReflFunc_bin{bin}", "[0]*(exp(-0.5*((x-[1])/[2])^2) + [3]*exp(-0.5*((x-[4])/[5])^2))", self.massRange[0], self.massRange[1])
        dataReflNorm = (fitFunc.GetParameter("reflectedRatio")*fitFunc.GetParameter("Amp")*fitFunc.GetParameter("Sigma")) / (fitFunc.GetParameter("Sigma1") + fitFunc.GetParameter("Frac2")*fitFunc.GetParameter("Sigma2"))
        dataReflFunc.SetParameters(dataReflNorm, fitFunc.GetParameter("Mean1"), fitFunc.GetParameter("Sigma1"), fitFunc.GetParameter("Frac2"), fitFunc.GetParameter("Mean2"), fitFunc.GetParameter("Sigma2"))
        dataReflFunc.SetLineColor(r.kGreen - 1)
        dataReflFunc.SetLineStyle(r.kDashed)

        self.ptBins[bin].fitFunc = fitFunc
        self.ptBins[bin].signalFunc = signalFunc
        self.ptBins[bin].backgroundFunc = backgroundFunc
        self.ptBins[bin].dataReflFunc = dataReflFunc
        self.ptBins[bin].create_fit_results(self.massRange[0], self.massRange[1])

    def fit_inv_mass_normsum(self, bin, backgroundName, fitRange = [1.64, 2.08], initialParams = []):
        self.ptBins[bin].fitRange = fitRange
        print(f"====== Fitting bin {bin} ({self.ptBins[bin].lowerPt} < pT < {self.ptBins[bin].upperPt} GeV/c) ======")
        # Fit the reflected MC histogram with a double Gaussian
        reflFunc = r.TF1(f"fDoubleGauss_bin{bin}", "[0]*exp(-0.5*((x-[1])/[2])^2) + [3]*exp(-0.5*((x-[4])/[5])^2)", 1.3, 2.3)
        reflFunc.SetParameters(1, self.ptBins[bin].massPtSliceReflected.GetMean(), self.ptBins[bin].massPtSliceReflected.GetRMS()/2, 1, self.ptBins[bin].massPtSliceReflected.GetMean(), self.ptBins[bin].massPtSliceReflected.GetRMS()/2)
        reflFunc.SetParNames("Amp1", "Mean1", "Sigma1", "Amp2", "Mean2", "Sigma2")
        # Require the amplitudes to be positive
        reflFunc.SetParLimits(0, 0, 1e6)
        reflFunc.SetParLimits(3, 0, 1e6)
        print("---- Fitting MC matched, reflected ----")
        self.ptBins[bin].massPtSliceReflected.Fit(reflFunc, "LR0")
        self.ptBins[bin].reflFunc = reflFunc

        self.ptBins[bin].reflectedRatio = self.ptBins[bin].massPtSliceReflected.GetEntries() / self.ptBins[bin].nMatched

        # Set up fitting function
        signalAndReflBkgFunc = r.TF1(f"signalAndReflBkgFunc_bin{bin}", "exp(-0.5*((x-[0])/[1])^2) + [2]*(exp(-0.5*((x-[3])/[4])^2) + [5]*exp(-0.5*((x-[6])/[7])^2))", fitRange[0], fitRange[1])
        signalAndReflBkgFunc.SetParNames("Mean", "Sigma", "ReflRatio", "Mean1", "Sigma1", "Frac2", "Mean2", "Sigma2")
        signalAndReflBkgFunc.SetParameter("Mean", 1.85)
        signalAndReflBkgFunc.SetParameter("Sigma", 0.12)
        if backgroundName == "pol2":
            combBkgFunc = r.TF1(f"combBkgPol2_{bin}", "([0]*x*x + [1]*x + 1)", fitRange[0], fitRange[1])
            combBkgFunc.SetParNames("a", "b")
            if len(initialParams) == 0:
                # Find good initial parameters
                x0 = fitRange[0]
                y0 = self.ptBins[bin].massPtSlice.GetBinContent(self.ptBins[bin].massPtSlice.GetXaxis().FindBin(x0))
                x1 = fitRange[1]
                y1 = self.ptBins[bin].massPtSlice.GetBinContent(self.ptBins[bin].massPtSlice.GetXaxis().FindBin(x1))
                b = (y1 - y0) / (x1 - x0)
                c = y0 - b*x0
                combBkgFunc.SetParameters(0, b/c) 
                fitNormSum = r.TF1NormSum(signalAndReflBkgFunc, combBkgFunc, c/100, c)
            else:
                raise Exception("Custom intial values are not implemented yet!")

            fitFunc = r.TF1(f"fGaussPol2_bin{bin}", fitNormSum, fitRange[0], fitRange[1], fitNormSum.GetNpar())

        elif backgroundName == "exp":
            combBkgFunc = r.TF1(f"combBkgExp_{bin}", "exp([0]*x)", fitRange[0], fitRange[1])
            combBkgFunc.SetParNames("A")
            if len(initialParams) == 0:
                combBkgFunc.SetParameters(-2)
                fitNormSum = r.TF1NormSum(signalAndReflBkgFunc, combBkgFunc, 25, 500)
            else:
                raise Exception("Custom intial values are not implemented yet!")

            fitFunc = r.TF1(f"fGaussExp_bin{bin}", fitNormSum, fitRange[0], fitRange[1], fitNormSum.GetNpar())

        else:
            raise Exception(f"Invalid background function '{backgroundName}'")

        fitFunc.SetParameters(np.asarray(fitNormSum.GetParameters()))

        fitFunc.SetParName(0, "normSignal")
        fitFunc.SetParName(1, "normCombBkg")
        for i in range(2, fitFunc.GetNpar()):
            fitFunc.SetParName(i, fitNormSum.GetParName(i))

        fitFunc.SetParameter(4, self.ptBins[bin].reflectedRatio)
        fitFunc.SetParameter(5, reflFunc.GetParameter("Mean1"))
        fitFunc.SetParameter(6, reflFunc.GetParameter("Sigma1"))
        fitFunc.SetParameter(7, reflFunc.GetParameter("Amp2")/reflFunc.GetParameter("Amp1"))
        fitFunc.SetParameter(8, reflFunc.GetParameter("Mean2"))
        fitFunc.SetParameter(9, reflFunc.GetParameter("Sigma2"))
        fitFunc.FixParameter(4, self.ptBins[bin].reflectedRatio)
        fitFunc.FixParameter(5, reflFunc.GetParameter("Mean1"))
        fitFunc.FixParameter(6, reflFunc.GetParameter("Sigma1"))
        fitFunc.FixParameter(7, reflFunc.GetParameter("Amp2")/reflFunc.GetParameter("Amp1"))
        fitFunc.FixParameter(8, reflFunc.GetParameter("Mean2"))
        fitFunc.FixParameter(9, reflFunc.GetParameter("Sigma2"))

        # Fit the histogram
        print("---- Fitting data ----")
        # self.ptBins[bin].massPtSlice.Fit(fitFunc, "LR0")

        # Obtain the signal and background functions separately
        signalFunc = r.TF1(f"signalFunc_bin{bin}", "[0]*exp(-0.5*((x-[1])/[2])^2)", fitRange[0], fitRange[1])
        signalFunc.SetParameters(fitFunc.GetParameter("normSignal"), fitFunc.GetParameter("Mean"), fitFunc.GetParameter("Sigma"))

        if backgroundName == "pol2":
            backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", "[0]*([1]*x*x + [2]*x + 1)", fitRange[0], fitRange[1])
            backgroundFunc.SetParameters(fitFunc.GetParameter("nCombBkg"), fitFunc.GetParameter("a"), fitFunc.GetParameter("b"))
        elif backgroundName == "exp":
            backgroundFunc = r.TF1(f"backgroundFunc_bin{bin}", "([0]*exp([1]*x))", fitRange[0], fitRange[1])
            backgroundFunc.SetParameters(fitFunc.GetParameter("nCombBkg"), fitFunc.GetParameter("A"))
        else:
            raise Exception(f"Background function '{backgroundName}' not implemented when extracting separate background shape!")
        backgroundFunc.SetLineColor(r.kBlue)

        dataReflFunc = r.TF1(f"dataReflFunc_bin{bin}", "[0]*(exp(-0.5*((x-[1])/[2])^2) + [3]*exp(-0.5*((x-[4])/[5])^2))", fitRange[0], fitRange[1])
        dataReflNorm = fitFunc.GetParameter("ReflRatio")*fitFunc.GetParameter("normSignal")
        dataReflFunc.SetParameters(dataReflNorm, fitFunc.GetParameter("Mean1"), fitFunc.GetParameter("Sigma1"), fitFunc.GetParameter("Frac2"), fitFunc.GetParameter("Mean2"), fitFunc.GetParameter("Sigma2"))
        dataReflFunc.SetLineColor(r.kGreen - 1)
        dataReflFunc.SetLineStyle(r.kDashed)

        self.ptBins[bin].fitFunc = fitFunc
        self.ptBins[bin].signalFunc = signalFunc
        self.ptBins[bin].backgroundFunc = backgroundFunc
        self.ptBins[bin].dataReflFunc = dataReflFunc
        self.ptBins[bin].create_fit_results()

    def calculate_correction(self, reweighting = True):
        self.reweighting = reweighting
        self.effRecOverGenFine = self.histD0PtMatched.Clone()
        self.effRecOverGenFine.Divide(self.histD0PtGenerated)
        print(f"Rec/Gen histogram was calculated using '{self.groupNameD0PtMatched}' and '{self.groupNameD0Generated}'")
        self.effRecOverGenFine.SetName("effRecOverGenFine")
        self.effRecOverGenFine.SetTitle("Reconstructed, matched D0 / Generated D0 after BC cuts")
        # Make sure the pT axis is (0, 12) GeV
        if (self.effRecOverGenFine.FindBin(self.ptBinsArray[-1]) <= self.effRecOverGenFine.GetNbinsX()):
            nFineBins = self.effRecOverGenFine.FindBin(self.ptBinsArray[-1])
            histTemp = r.TH1F("effRecOverGenFine", "Reconstructed, matched D0 / Generated D0 after BC cuts", nFineBins, self.ptBinsArray[0], self.ptBinsArray[-1])
            for i in range(1, nFineBins + 1):
                histTemp.SetBinContent(i, self.effRecOverGenFine.GetBinContent(i))
                histTemp.SetBinError(i, self.effRecOverGenFine.GetBinError(i))

            self.effRecOverGenFine = histTemp
        # Create a new version of the correction factor hist with the final binning and range
        self.histD0PtMatchedFinalBins = self.histD0PtMatched.Rebin(len(self.ptBinsArray) - 1, f"PtMcMatchedFinalBins", np.asarray(self.ptBinsArray, 'd'))
        self.histD0PtGeneratedFinalBins = self.histD0PtGenerated.Rebin(len(self.ptBinsArray) - 1, f"projPtMcGenFinalBins_y_{self.minY}_{self.maxY}", np.asarray(self.ptBinsArray, 'd'))
        self.effRecOverGen = self.histD0PtMatchedFinalBins.Clone()
        self.effRecOverGen.SetName("effRecOverGen")
        self.effRecOverGen.SetTitle("Reconstructed / Generated;p_{T};Ratio")
        self.effRecOverGen.Divide(self.histD0PtGeneratedFinalBins)
        # Apply correction factor
        self.histCorrectedSpectrum = self.histRawYield.Clone()
        self.histCorrectedSpectrum.SetName("histCorrectedSpectrumWithoutReweighting")
        self.histCorrectedSpectrum.SetTitle("Corrected pT spectrum without reweighting")
        self.histCorrectedSpectrum.GetYaxis().SetTitle("dN/dp_{T}")
        self.histCorrectedSpectrum.Divide(self.effRecOverGen)

        if reweighting:
            self.reweightingFunc = r.TF1("powerLaw", "[0]*x/TMath::Power((1+TMath::Power(x/[1],[3])),[2])", 0, 12)
            self.reweightingFunc.SetParameters(394000, 1.83, 1.78, 2.87)
            print("===== Fitting to the corrected spectrum for reweighting =====")
            self.reweightingFitResults = self.histCorrectedSpectrum.Fit(self.reweightingFunc, "LS0")
            self.effRecOverGenReweighted = self.effRecOverGen.Clone()
            self.effRecOverGenReweighted.Reset()
            self.effRecOverGenReweighted.SetName("effRecOverGenReweighted")
            self.effRecOverGenReweighted.SetTitle("Reconstructed / Generated, reweighted")
            dpT = self.effRecOverGenFine.GetBinWidth(1)
            # Calculate numerator of <epsilon>_i
            for i in range(1, self.effRecOverGenFine.GetNbinsX() + 1):
                binCenter = self.effRecOverGenFine.GetBinCenter(i)
                binContent = self.effRecOverGenFine.GetBinContent(i)
                binError = self.effRecOverGenFine.GetBinError(i)
                weight = self.reweightingFunc.Eval(binCenter)
                # Calculate weighted content
                weightedContent = binContent * weight * dpT
                weightedError = np.abs(weight) * dpT * binError # Error propagation
                # Find the target bin and fill it
                targetBin = self.effRecOverGenReweighted.FindBin(binCenter)
                self.effRecOverGenReweighted.AddBinContent(targetBin, weightedContent)
                currentErrorTarget = self.effRecOverGenReweighted.GetBinError(targetBin)
                self.effRecOverGenReweighted.SetBinError(targetBin, np.sqrt(currentErrorTarget**2 + weightedError**2))
            # Calculate denominator of <epsilon>_i
            histDenominator = self.effRecOverGen.Clone()
            histDenominator.Reset()
            # For some reason the first iteration of IntegralError gives a nonsensical value, throw this away first
            _ = self.reweightingFunc.IntegralError(0., 0.1, self.reweightingFitResults.GetParams(), self.reweightingFitResults.GetCovarianceMatrix().GetMatrixArray(), epsilon=1e-8)
            for i in range(1, self.effRecOverGenReweighted.GetNbinsX() + 1):
                lowEdge = self.effRecOverGenReweighted.GetXaxis().GetBinLowEdge(i)
                upEdge = self.effRecOverGenReweighted.GetXaxis().GetBinUpEdge(i)
                print(f"--- Calculating integral of reweightingFunc from {lowEdge} to {upEdge} ---")
                integral = self.reweightingFunc.Integral(lowEdge, upEdge)
                histDenominator.SetBinContent(i, integral)
                integralError = self.reweightingFunc.IntegralError(lowEdge, upEdge, self.reweightingFitResults.GetParams(), self.reweightingFitResults.GetCovarianceMatrix().GetMatrixArray(), epsilon=1e-8)
                print(f"Bin {i}: integral = {integral} +- {integralError}")
                histDenominator.SetBinError(i, integralError)
            # Obtain <epsilon>_i as a histogram
            self.effRecOverGenReweighted.Divide(histDenominator)
            self.effRecOverGenWithoutReweighting = self.effRecOverGen
            self.effRecOverGenWithoutReweighting.SetLineColor(r.kRed)
            self.effRecOverGenWithoutReweighting.SetTitle("Reconstructed / Generated, without reweighting")
            self.effRecOverGen = self.effRecOverGenReweighted

            # Apply reweighted correction factor
            self.histCorrectedSpectrumReweighted = self.histRawYield.Clone()
            self.histCorrectedSpectrumReweighted.SetName("histCorrectedSpectrumWithReweighting")
            self.histCorrectedSpectrumReweighted.SetTitle("Corrected pT spectrum with reweighting")
            self.histCorrectedSpectrumReweighted.GetYaxis().SetTitle("dN/dp_{T}")
            self.histCorrectedSpectrumReweighted.Divide(self.effRecOverGen)
            self.histCorrectedSpectrumWithoutReweighting = self.histCorrectedSpectrum
            self.histCorrectedSpectrumWithoutReweighting.SetLineColor(r.kRed)
            self.histCorrectedSpectrumWithoutReweighting.SetTitle("Corrected spectrum, without reweighting")
            self.histCorrectedSpectrum = self.histCorrectedSpectrumReweighted

    def calculate_isitsupcmode_efficiency(self):
        if self.isITSUPCMode == 2:
            raise Exception("Analysis was not initialized with an IsITSUPCMode value specified!")
        # Calculate the efficiency of the selection applied on IsITSUPCMode, which is to be multiplied by the correction factor obtained in calculate_correction
        if self.old:
            histD0MassPtIsITSUPCMode = self.fileTableReader.Get("analysis-asymmetric-pairing/output").FindObject("PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepTauxyzprojCut").FindObject("MyMassPtIsITSUPCModeHisto")
        else:
            histD0MassPtIsITSUPCMode = self.fileTableReader.Get("analysis-asymmetric-pairing/output").FindObject("PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_D0StrictTopoCuts2").FindObject("MyMassPtIsITSUPCModeHisto")
        # Select D0 candidates by projecting out the D0 mass range (TODO: use a histogram with a very good S/B for this specific purpose)
        lowerBin = histD0MassPtIsITSUPCMode.GetXaxis().FindBin(1.8) + 1
        upperBin = histD0MassPtIsITSUPCMode.GetXaxis().FindBin(1.9)
        histD0MassPtIsITSUPCMode.GetXaxis().SetRange(lowerBin, upperBin)
        histD0PtIsITSUPCMode = histD0MassPtIsITSUPCMode.Project3D("zy")
        self.histD0PtIsITSUPCModeAll = histD0PtIsITSUPCMode.ProjectionX("histD0PtIsITSUPCModeAll", 1, 2)
        self.histD0PtIsITSUPCModeAll = self.histD0PtIsITSUPCModeAll.Rebin(len(self.ptBinsArray) - 1, "histD0PtIsITSUPCModeAll", np.asarray(self.ptBinsArray, 'd'))
        self.histD0PtIsITSUPCModeSelected = histD0PtIsITSUPCMode.ProjectionX(f"histD0PtIsITSUPCMode{self.isITSUPCMode}", self.isITSUPCMode + 1, self.isITSUPCMode + 1)
        self.histD0PtIsITSUPCModeSelected = self.histD0PtIsITSUPCModeSelected.Rebin(len(self.ptBinsArray) - 1, "histD0PtIsITSUPCModeSelected", np.asarray(self.ptBinsArray, 'd'))
        self.histIsITSUPCModeEfficiency = self.histD0PtIsITSUPCModeSelected.Clone()
        self.histIsITSUPCModeEfficiency.SetName("histIsITSUPCModeEfficiency")
        self.histIsITSUPCModeEfficiency.SetTitle(f"IsITSUPCMode={self.isITSUPCMode} selection efficiency (est. from data)")
        self.histIsITSUPCModeEfficiency.Divide(self.histD0PtIsITSUPCModeAll)

    def calculate_partial_efficiencies(self):
        """
        Calculate factorized efficiencies for some predefined (hardcoded) factorizations
        Total efficiency = N(rec. matched D0 after all cuts) / N(gen. D0 after BC cuts)
        """
        # Prepare histograms
        if not hasattr(self, 'histD0PtGeneratedFinalBins'):
            self.histD0PtGeneratedFinalBins = self.histD0PtGenerated.Rebin(len(self.ptBinsArray) - 1, f"projPtMcGenFinalBins_y_{self.minY}_{self.maxY}", np.asarray(self.ptBinsArray, 'd'))

        self.histD0PtYGenInRecEvent = self.fileGen.Get("analysis-asymmetric-pairing/output").FindObject("MCTruthGenRec_D0FS").FindObject("MyMcPtYHisto")
        lowerYBin = self.histD0PtYGenInRecEvent.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenInRecEvent.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenInRecEvent = self.histD0PtYGenInRecEvent.ProjectionX(f"projPtMcGenInRecEvent_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenInRecEvent.SetTitle(f"Generated D0 in reconstructed event, {self.minY} < y < {self.maxY}")
        self.histD0PtGenInRecEvent = self.histD0PtGenInRecEvent.Rebin(len(self.ptBinsArray) - 1, f"PtMcGenInRecEventFinalBins", np.asarray(self.ptBinsArray, 'd'))

        self.histD0PtYGenInSelEvent = self.fileGen.Get("analysis-asymmetric-pairing/output").FindObject("MCTruthGenSel_D0FS").FindObject("MyMcPtYHisto")
        lowerYBin = self.histD0PtYGenInSelEvent.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenInSelEvent.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenInSelEvent = self.histD0PtYGenInSelEvent.ProjectionX(f"projPtMcGenInSelEvent_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenInSelEvent.SetTitle(f"Generated D0 in reconstructed event, {self.minY} < y < {self.maxY}")
        self.histD0PtGenInSelEvent = self.histD0PtGenInSelEvent.Rebin(len(self.ptBinsArray) - 1, f"PtMcGenInSelEventFinalBins", np.asarray(self.ptBinsArray, 'd'))

        self.histD0PtMatchedInSelEvent = self.fileRec.Get("analysis-asymmetric-pairing/output").FindObject("PairsBarrelSEPM_noTrackCut:noTrackCut_KPiFromD0").FindObject("Pt")
        self.histD0PtMatchedInSelEvent = self.histD0PtMatchedInSelEvent.Rebin(len(self.ptBinsArray) - 1, f"PtMcMatchedInSelEvent", np.asarray(self.ptBinsArray, 'd'))
        self.histD0PtMatchedInSelEvent.SetTitle("Reconstructed, matched D0->Kpi in selected event, no track or pair cuts")

        self.histD0PtMatchedInSelEventAfterTrackCuts = self.fileRec.Get("analysis-asymmetric-pairing/output").FindObject("PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_KPiFromD0").FindObject("Pt")
        self.histD0PtMatchedInSelEventAfterTrackCuts = self.histD0PtMatchedInSelEventAfterTrackCuts.Rebin(len(self.ptBinsArray) - 1, f"PtMcMatchedInSelEventAfterTrackCuts", np.asarray(self.ptBinsArray, 'd'))
        self.histD0PtMatchedInSelEventAfterTrackCuts.SetTitle("Reconstructed, matched D0->Kpi in selected event, selected tracks, no pair cuts")

        if not hasattr(self, 'histD0PtMatchedFinalBins'):
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts = self.fileRec.Get("analysis-asymmetric-pairing/output").FindObject("PairsBarrelSEPM_kaonPIDTPCTOFpTDCAz:pionNoPIDpTDCAz_PtDepTauxyzprojCut_KPiFromD0").FindObject("Pt")
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts = self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts.Rebin(len(self.ptBinsArray) - 1, "PtMcMatchedInSelEventAfterTrackCutsAndPairCuts", np.asarray(self.ptBinsArray, 'd'))
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts.SetTitle("Reconstructed, matched D0->Kpi in selected event, selected tracks, selected pairs")
        else:
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts = self.histD0PtMatchedFinalBins.Clone()
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts.SetName("PtMcMatchedInSelEventAfterTrackCutsAndPairCuts")
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts.SetTitle("Reconstructed, matched D0->Kpi in selected event, selected tracks, selected pairs")

        # List to hold factorized efficiencies
        self.factorizedEfficiencies = []

        # Calculate efficiencies
        eff = FactorizedEfficiency(2)
        eff.add_factor("Reconstructed, matched D0 after all cuts", "MC gen D0 in reconstructed events", self.histD0PtMatchedFinalBins, self.histD0PtGenInRecEvent)
        eff.add_factor("MC gen D0 in reconstructed events", "MC gen D0 in all events passing BC cuts", self.histD0PtGenInRecEvent, self.histD0PtGeneratedFinalBins)
        eff.calculate_total_efficiency()
        self.factorizedEfficiencies.append(eff)
        del eff

        eff = FactorizedEfficiency(5)
        eff.add_factor("Gen. D0 in rec. evt.", "Gen. D0 in lumi", self.histD0PtGenInRecEvent, self.histD0PtGeneratedFinalBins)
        eff.add_factor("Gen. D0 in sel. evt.", "Gen. D0 in rec. evt.", self.histD0PtGenInSelEvent, self.histD0PtGenInRecEvent)
        eff.add_factor("Rec. matched D0 in sel. evt", "Gen. D0 in sel. evt.", self.histD0PtMatchedInSelEvent, self.histD0PtGenInSelEvent)
        eff.add_factor("Rec. matched D0 in sel. evt. after track cuts", "Rec. matched D0 in sel. evt.", self.histD0PtMatchedInSelEventAfterTrackCuts, self.histD0PtMatchedInSelEvent)
        eff.add_factor("Rec. matched D0 in sel. evt. after track and pair cuts", "Rec. matched D0 in sel. evt. after track cuts", self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts, self.histD0PtMatchedInSelEventAfterTrackCuts)
        eff.calculate_total_efficiency()
        self.factorizedEfficiencies.append(eff)
        del eff

    def create_raw_yield_histogram(self):
        self.histRawYield = r.TH1F("histRawYield", "Raw yield /#Delta p_{T}, raw stat. errors", len(self.ptBins), np.asarray(self.ptBinsArray, 'd'))
        self.histRawYield.GetYaxis().SetTitle("Raw D^{0} yield (1/GeV c^{-1})")
        for i, bin in enumerate(self.ptBins):
            self.histRawYield.SetBinContent(i+1, bin.nSignal / self.histRawYield.GetBinWidth(i+1))
            # Error propagation with the bin width
            self.histRawYield.SetBinError(i+1, bin.relativeStatError / self.histRawYield.GetBinWidth(i+1) * bin.nSignal)

    def calculate_spectrum_per_event(self):
        histEventsAfterCuts = self.fileTableReader.Get("analysis-event-selection/output").FindObject("Event_AfterCuts").FindObject("VtxZ")
        self.nEvents = histEventsAfterCuts.GetEntries()
        self.histCorrectedSpectrumPerEvent = self.histCorrectedSpectrum.Clone()
        self.histCorrectedSpectrumPerEvent.SetName("histCorrectedSpectrumPerEvent")
        self.histCorrectedSpectrumPerEvent.SetTitle("Corrected pT spectrum normalized by number of events")
        self.histCorrectedSpectrumPerEvent.Scale(1. / (self.maxY - self.minY)) # Divide by dy
        self.histCorrectedSpectrumPerEvent.Scale(1. / self.nEvents) # Normalize by number of events
        self.histCorrectedSpectrumPerEvent.GetYaxis().SetTitle("1/N_{events} d^{2}N/dp_{T}dy")
        self.canvasPtSpectrumPerEvent = r.TCanvas("canvasPtSpectrumPerEvent")
        self.canvasPtSpectrumPerEvent.cd()
        self.histCorrectedSpectrumPerEvent.Draw()
        r.gPad.SetLogy()
        self.canvasPtSpectrumPerEvent.Draw()

    def calculate_raw_yield_per_lumi(self):
        self.histRawYieldPerLumi = self.histRawYield.Clone()
        self.histRawYieldPerLumi.SetName("histRawYieldPerLumi")
        self.histRawYieldPerLumi.SetTitle("Raw yield /#Delta p_{T} L_{int}")
        self.histRawYieldPerLumi.Scale(1 / self.lumi) # µb / GeVc^-1
        self.histRawYieldPerLumi.Scale(1 / 1000.) # mb / GeVc^-1
        self.histRawYieldPerLumi.GetYaxis().SetTitle("Raw D^{0} yield / L_{int} (1/GeV c^{-1})")

    def calculate_cross_section(self):
        # Get the branching fraction from the PDG
        import pdg
        pdgApi = pdg.connect()
        pdgD0 = pdgApi.get_particle_by_name('D0')
        pdgD0KpiDecay = pdgApi.get(f'{pdgD0.baseid}.1/2025')
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
        self.canvasCrossSection = r.TCanvas("canvasCrossSection")
        self.canvasCrossSection.cd()
        self.histCrossSection.Draw()
        r.gPad.SetLogy()
        self.canvasCrossSection.Draw()

    def draw_efficiency(self):
        self.canvasEfficiency = r.TCanvas("canvasEfficiency", "canvasEfficiency", 1200, 666)
        self.canvasEfficiency.Divide(3, 2, 0.002, 0.01)
        self.canvasEfficiency.cd(1)
        self.histD0PtGenerated.Draw()
        self.canvasEfficiency.cd(2)
        self.histD0PtMatched.Draw()
        self.canvasEfficiency.cd(3)
        self.effRecOverGenFine.Draw()
        self.canvasEfficiency.cd(4)
        self.effRecOverGen.SetStats(0)
        self.effRecOverGen.Draw()
        if self.reweighting:
            self.effRecOverGenWithoutReweighting.SetStats(0)
            self.effRecOverGenWithoutReweighting.Draw("same")
            self.legendEfficiency4 = r.TLegend(0.55, 0.15, 0.9, 0.3)
            self.legendEfficiency4.AddEntry(self.effRecOverGen, "With reweighting")
            self.legendEfficiency4.AddEntry(self.effRecOverGenWithoutReweighting, "Without reweighting")
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


    def draw_fits_and_yield(self):
        # Figure out grid layout
        nPanels = len(self.ptBins) + 1
        nRows = int(np.ceil(nPanels/3))
        self.canvasFitsYields = r.TCanvas("canvasFitsYields", "canvasFitsYields", 1200, nRows * 333)
        self.canvasFitsYields.Divide(3, nRows, 0.002, 0.01)

        self.textBoxesFitsYields = []
        for i, bin in enumerate(self.ptBins):
            self.histRawYield.SetBinContent(i+1, bin.nSignal / self.histRawYield.GetBinWidth(i+1))
            # Error propagation with the bin width
            self.histRawYield.SetBinError(i+1, bin.relativeStatError / self.histRawYield.GetBinWidth(i+1) * bin.nSignal)
            # Draw the histograms with fits
            self.canvasFitsYields.cd(i+1)
            bin.massPtSlice.Draw("E")
            bin.massPtSlice.SetStats(0)
            bin.fitFunc.Draw("same")
            bin.backgroundFunc.Draw("same")
            bin.totalBackgroundFunc = r.TF1(f"totalBackgroundFunc_bin{bin}", str(bin.backgroundFunc.GetExpFormula("p")) + "+" + str(bin.dataReflFunc.GetExpFormula("p")), self.massRange[0], self.massRange[1])
            bin.totalBackgroundFunc.SetLineColor(r.kGreen)
            bin.totalBackgroundFunc.SetLineStyle(r.kDashed)
            bin.totalBackgroundFunc.Draw("same")
            # Text box with info
            self.textBoxesFitsYields.append(r.TPaveText(0.65, 0.5, 0.95, 0.85, "NDC"))
            self.textBoxesFitsYields[i].SetName(f"textbox_bin{i}")
            self.textBoxesFitsYields[i].SetFillColor(0)
            self.textBoxesFitsYields[i].SetFillStyle(0)
            self.textBoxesFitsYields[i].SetBorderSize(0)
            self.textBoxesFitsYields[i].SetTextAlign(12)
            self.textBoxesFitsYields[i].SetTextFont(42)
            self.textBoxesFitsYields[i].SetTextSize(0.04)
            self.textBoxesFitsYields[i].AddText(f"#mu = {bin.fitMu:.3f}")
            self.textBoxesFitsYields[i].AddText(f"#sigma = {bin.fitSigma:.3f}")
            self.textBoxesFitsYields[i].AddText(f"S = {bin.nSignal:.3f}")
            self.textBoxesFitsYields[i].AddText(f"B = {bin.nBackground:.3f}")
            self.textBoxesFitsYields[i].AddText(f"S/#sqrt{{S+B}} = {bin.nSignal/(np.sqrt(bin.nSignal + bin.nBackground)):.3f}")
            self.textBoxesFitsYields[i].AddText(f"Refl/S = {bin.nReflBackground/bin.nSignal:.3f}")
            self.textBoxesFitsYields[i].AddText(f"#chi^{{2}}/ndf = {bin.fitChi2Ndf:.3f}")
            self.textBoxesFitsYields[i].Draw()

        # Draw raw yield histogram
        self.canvasFitsYields.cd(nPanels)
        self.histRawYield.Draw()

        self.canvasFitsYields.Draw()

    def draw_reflected_fits(self):
        # Figure out grid layout
        nPanels = len(self.ptBins)
        nRows = int(np.ceil(nPanels/3))
        self.canvasReflFits = r.TCanvas("canvasReflFits", "canvasReflFits", 1200, nRows * 333)
        self.canvasReflFits.Divide(3, nRows, 0.002, 0.01)

        self.textBoxesReflected = []
        for i, bin in enumerate(self.ptBins):
            self.canvasReflFits.cd(i+1)
            bin.massPtSliceReflected.Draw("E")
            bin.reflFunc1 = r.TF1(f"fReflGauss1_bin{bin}", "[0]*exp(-0.5*((x-[1])/[2])^2)", 1.3, 2.3)
            bin.reflFunc1.SetParameters(bin.reflFunc.GetParameter("Amp1"), bin.reflFunc.GetParameter("Mean1"), bin.reflFunc.GetParameter("Sigma1"))
            bin.reflFunc1.SetLineColor(r.kGreen)
            bin.reflFunc1.SetLineStyle(r.kDashed)
            bin.reflFunc1.Draw("same")
            bin.reflFunc2 = r.TF1(f"fReflGauss2_bin{bin}", "[0]*exp(-0.5*((x-[1])/[2])^2)", 1.3, 2.3)
            bin.reflFunc2.SetParameters(bin.reflFunc.GetParameter("Amp2"), bin.reflFunc.GetParameter("Mean2"), bin.reflFunc.GetParameter("Sigma2"))
            bin.reflFunc2.SetLineColor(r.kBlue)
            bin.reflFunc2.SetLineStyle(r.kDashed)
            bin.reflFunc2.Draw("same")
            bin.reflFuncSum = r.TF1(f"fReflGauss1Gauss2_bin{bin}", "[0]*exp(-0.5*((x-[1])/[2])^2) + [3]*exp(-0.5*((x-[4])/[5])^2)", 1.3, 2.3)
            bin.reflFuncSum.SetParameters(np.asarray(bin.reflFunc.GetParameters()))
            bin.reflFuncSum.SetLineColor(r.kRed)
            bin.reflFuncSum.Draw("same")
            # Text box with info
            self.textBoxesReflected.append(r.TPaveText(0.6, 0.4, 1., 0.9, "NDC"))
            self.textBoxesReflected[i].SetName(f"textboxreflected_bin{i}")
            self.textBoxesReflected[i].SetFillColor(0)
            self.textBoxesReflected[i].SetFillStyle(0)
            self.textBoxesReflected[i].SetBorderSize(0)
            self.textBoxesReflected[i].SetTextAlign(12)
            self.textBoxesReflected[i].SetTextFont(42)
            self.textBoxesReflected[i].SetTextSize(0.04)
            self.textBoxesReflected[i].AddText(f"Refl/S = {bin.reflectedRatio:.3f}")
            self.textBoxesReflected[i].AddText(f"chi2/ndof = {bin.reflChi2Ndf:.3f}")
            self.textBoxesReflected[i].AddText(f"Amp1 = {bin.reflFunc.GetParameter('Amp1'):.3f}")
            self.textBoxesReflected[i].AddText(f"Mean1 = {bin.reflFunc.GetParameter('Mean1'):.3f}")
            self.textBoxesReflected[i].AddText(f"Sigma1 = {bin.reflFunc.GetParameter('Sigma1'):.3f}")
            self.textBoxesReflected[i].AddText(f"Amp2 = {bin.reflFunc.GetParameter('Amp2'):.3f}")
            self.textBoxesReflected[i].AddText(f"Mean2 = {bin.reflFunc.GetParameter('Mean2'):.3f}")
            self.textBoxesReflected[i].AddText(f"Sigma2 = {bin.reflFunc.GetParameter('Sigma2'):.3f}")
            self.textBoxesReflected[i].Draw()

        self.canvasReflFits.Draw()
