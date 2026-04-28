import ROOT as r
import numpy as np

background_npar = {'pol3pol2' : 6,
                   'pol3': 4,
                   'ratioPol2' : 6,
                   'pol2' : 3,
                   'exp' : 2}

def signal(x: np.ndarray, par: np.ndarray) -> float:
    mass = x[0]
    # integral = par[0]
    return (par[0] / (par[2] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((mass - par[1]) / par[2])**2)

def dscb(x: np.ndarray, par: np.ndarray) -> float:
    # Double sided crystal ball for fitting Jpsi->mumu
    N = par[0]
    m0 = par[1]
    sigma = par[2]
    alphaL = par[3]
    nL = par[4]
    alphaR = par[5]
    nR = par[6]
    AL = (nL/np.abs(alphaL))**nL * np.exp(-(np.abs(alphaL)**2)/2)
    AR = (nR/np.abs(alphaR))**nR * np.exp(-(np.abs(alphaR)**2)/2)
    BL = nL/np.abs(alphaL) - np.abs(alphaL)
    BR = nR/np.abs(alphaR) - np.abs(alphaR)
    t = (x[0] - m0) / sigma
    result = 0
    if (-alphaL <= t and alphaR >= t):
        result = np.exp(-(t**2)/2)
    elif (t < -alphaL):
        result = AL*(BL-t)**(-nL)
    elif (t > alphaR):
        result = AR*(BR+t)**(-nR)
    return N*result

def reflected_background(x: np.ndarray, par: np.ndarray) -> float:
    mass = x[0]
    """
    par[0] = integral
    par[1] = relative normalization of the two Gaussians
    par[2] = Gaussian 1 mu
    par[3] = Gaussian 1 sigma
    par[4] = Gaussian 2 mu
    par[5] = Gaussian 2 sigma
    """
    return par[0] * ( (par[1] / (par[3] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((mass - par[2]) / par[3])**2)
                     + ((1 - par[1]) / (par[5] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((mass - par[4]) / par[5])**2))
    
def pol3pol2(x: np.ndarray, par: np.ndarray) -> float:
    # pol3 / pol2
    mass = x[0]
    return (par[0] * mass**3 + par[1] * mass**2 + par[2] * mass + par[3]) / (par[4] * mass**2 + par[5] * mass + 1)
    # return par[0] * (1.0 + par[1] * mass + par[2] * mass**2 + par[3] * mass**3) / (1.0 + par[4] * mass + par[5] * mass**2)
    # Parameterized in terms of positions of local minimum and maximum for the numerator
    # return (par[0] * (mass**3/3 - (par[1]+par[2])*mass**2/2 + par[1]*par[2]*mass) + par[3]) / (1.0 + par[4] * mass + par[5] * mass**2)

def pol3(x: np.ndarray, par: np.ndarray) -> float:
    # pol3
    mass = x[0]
    # Parameterized in terms of positions of local minimum and maximum for the numerator
    # return par[0] * (mass**3/3 - (par[1]+par[2])*mass**2/2 + par[1]*par[2]*mass) + par[3]
    return par[0] * mass**3 + par[1] * mass**2 + par[2] * mass + par[3]

def ratioPol2(x: np.ndarray, par: np.ndarray) -> float:
    # pol2 / pol2
    mass = x[0]
    return (par[0] + par[1]*mass + par[2]*mass**2)/(par[3] + par[4]*mass + par[5]*mass**2)

def pol2(x: np.ndarray, par: np.ndarray) -> float:
    mass = x[0]
    return (par[0]*mass**2 + par[1]*mass + par[2])

def exp(x: np.ndarray, par: np.ndarray) -> float:
    mass = x[0]
    return par[0]*np.exp(par[1]*mass)

def simple_fit_model(x: np.ndarray, par: np.ndarray) -> float:
    mass = x[0]
    signalVal = (par[0] / (par[2] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((mass - par[1]) / par[2])**2)
    backgroundVal = par[3] + par[4] * mass
    return signalVal + backgroundVal

def template_shape(x: np.ndarray, par: np.ndarray, hist=None) -> float:
    # par[0] is the normalization, shape is fixed
    if hist is not None:
        return par[0] * hist.Interpolate(x)
    else:
        return 0.0

def make_fit_model(backgroundFunc, backgroundNPar, xmin, xmax):
    # Composite callable that TF1 will call
    def fit_model(x: np.ndarray, par: np.ndarray) -> float:
        # First 3 parameters are S, mu, sigma for the signal Gaussian
        signalVal = signal(x, [par[0], par[1], par[2]])
        # Next 6 parameters are for the reflected background
        # par[3] is the reflected/signal ratio, so it must be multiplied by S=par[0] before being passed
        reflVal = reflected_background(x, [par[0]*par[3], par[4], par[5], par[6], par[7], par[8]])
        # Next backgroundNPar parameters are for the combinatorial background
        backgroundPars = []
        for i in range(backgroundNPar):
            backgroundPars.append(par[3 + 6 + i])
        backgroundVal = backgroundFunc(x, backgroundPars)
        return signalVal + reflVal + backgroundVal
    totalFunc = r.TF1("fit_model", fit_model, xmin, xmax, 3 + 6 + backgroundNPar)
    totalFunc.SetParName(0, "S")
    totalFunc.SetParName(1, "Mean")
    totalFunc.SetParName(2, "Sigma")
    totalFunc.SetParName(3, "reflToSignalRatio")
    totalFunc.SetParName(4, "reflRelNorm")
    totalFunc.SetParName(5, "reflMu1")
    totalFunc.SetParName(6, "reflSigma1")
    totalFunc.SetParName(7, "reflMu2")
    totalFunc.SetParName(8, "reflSigma2")
    return totalFunc

def make_fit_model_with_corr_bkg(backgroundFunc, backgroundNPar, xmin, xmax, histCorrBkg=None):
    # Composite callable that TF1 will call
    def fit_model(x: np.ndarray, par: np.ndarray) -> float:
        # First 3 parameters are S, mu, sigma for the signal Gaussian
        signalVal = signal(x, [par[0], par[1], par[2]])
        # Next 6 parameters are for the reflected background
        # par[3] is the reflected/signal ratio, so it must be multiplied by S=par[0] before being passed
        reflVal = reflected_background(x, [par[0]*par[3], par[4], par[5], par[6], par[7], par[8]])
        # Next backgroundNPar parameters are for the combinatorial background
        backgroundPars = []
        for i in range(backgroundNPar):
            backgroundPars.append(par[3 + 6 + i])
        backgroundVal = backgroundFunc(x, backgroundPars)
        # Lastly, add the template value for the correlated background from D0 -> K- pi+ pi0 if requested
        corrBkgVal = 0
        if histCorrBkg is not None:
            # corrBkgVal = template_shape(x, [par[-1]])
            corrBkgVal = par[3 + 6 + backgroundNPar] * histCorrBkg.Interpolate(x[0])
        return signalVal + reflVal + backgroundVal + corrBkgVal
    totalFunc = r.TF1("fit_model", fit_model, xmin, xmax, 3 + 6 + backgroundNPar + 1)
    totalFunc.SetParName(0, "S")
    totalFunc.SetParName(1, "Mean")
    totalFunc.SetParName(2, "Sigma")
    totalFunc.SetParName(3, "reflToSignalRatio")
    totalFunc.SetParName(4, "reflRelNorm")
    totalFunc.SetParName(5, "reflMu1")
    totalFunc.SetParName(6, "reflSigma1")
    totalFunc.SetParName(7, "reflMu2")
    totalFunc.SetParName(8, "reflSigma2")
    if histCorrBkg is not None:
        # totalFunc._templateHist = histCorrBkg
        totalFunc.SetParName(3 + 6 + backgroundNPar, "nCorr")
    return totalFunc

def make_correlated_background_model(xmin, xmax, histCorrBkg):
    def corr_bkg_model(x: np.ndarray, par: np.ndarray) -> float:
        return par[0] * histCorrBkg.Interpolate(x[0])
    func = r.TF1("correlated_bkg_model", corr_bkg_model, xmin, xmax, 1)
    # func._templateHist = histCorrBkg
    return func

def make_jpsi_fit_model(backgroundFunc, backgroundNPar, xmin, xmax):
    # Composite callable that TF1 will call
    def fit_model(x: np.ndarray, par: np.ndarray) -> float:
        # First 3 parameters are S, mu, sigma for the signal Gaussian
        signalVal = dscb(x, [par[0], par[1], par[2], par[3], par[4], par[5], par[6]])
        # Last backgroundNPar parameters are for the combinatorial background
        backgroundPars = []
        for i in range(backgroundNPar):
            backgroundPars.append(par[7 + i])
        backgroundVal = backgroundFunc(x, backgroundPars)
        return signalVal + backgroundVal
    totalFunc = r.TF1("fit_model", fit_model, xmin, xmax, 7 + backgroundNPar)
    totalFunc.SetParName(0, "N")
    totalFunc.SetParName(1, "m0")
    totalFunc.SetParName(2, "sigma")
    totalFunc.SetParName(3, "alphaL")
    totalFunc.SetParName(4, "nL")
    totalFunc.SetParName(5, "alphaR")
    totalFunc.SetParName(6, "nR")
    return totalFunc

def make_background_model(backgroundFunc, backgroundNPar, xmin, xmax):
    # Composite callable that TF1 will call
    def background_model(x: np.ndarray, par: np.ndarray) -> float:
        # First 6 parameters are for the reflected background
        reflVal = reflected_background(x, [par[0], par[1], par[2], par[3], par[4], par[5]])
        # Last backgroundNPar parameters are for the combinatorial background
        backgroundPars = []
        for i in range(backgroundNPar):
            backgroundPars.append(par[6 + i])
        backgroundVal = backgroundFunc(x, backgroundPars)
        return reflVal + backgroundVal
    totalFunc = r.TF1("background_model", background_model, xmin, xmax, 6 + backgroundNPar)
    totalFunc.SetParName(0, "reflN")
    totalFunc.SetParName(1, "reflRelNorm")
    totalFunc.SetParName(2, "reflMu1")
    totalFunc.SetParName(3, "reflSigma1")
    totalFunc.SetParName(4, "reflMu2")
    totalFunc.SetParName(5, "reflSigma2")
    return totalFunc

def make_background_model_with_corr_bkg(backgroundFunc, backgroundNPar, xmin, xmax, histCorrBkg):
    # Composite callable that TF1 will call
    def background_model(x: np.ndarray, par: np.ndarray) -> float:
        # First 6 parameters are for the reflected background
        reflVal = reflected_background(x, [par[0], par[1], par[2], par[3], par[4], par[5]])
        # Last backgroundNPar parameters are for the combinatorial background
        backgroundPars = []
        for i in range(backgroundNPar):
            backgroundPars.append(par[6 + i])
        backgroundVal = backgroundFunc(x, backgroundPars)
        corrBkgVal = par[3 + 6 + backgroundNPar] * histCorrBkg.Interpolate(x[0])
        return reflVal + backgroundVal + corrBkgVal
    totalFunc = r.TF1("background_model", background_model, xmin, xmax, 6 + backgroundNPar + 1)
    totalFunc.SetParName(0, "reflN")
    totalFunc.SetParName(1, "reflRelNorm")
    totalFunc.SetParName(2, "reflMu1")
    totalFunc.SetParName(3, "reflSigma1")
    totalFunc.SetParName(4, "reflMu2")
    totalFunc.SetParName(5, "reflSigma2")
    # totalFunc._templateHist = histCorrBkg
    totalFunc.SetParName(3 + 6 + backgroundNPar, "nCorr")
    return totalFunc
