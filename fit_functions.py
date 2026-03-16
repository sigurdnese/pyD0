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

def make_fit_model(backgroundFunc, backgroundNPar, xmin, xmax):
    # Composite callable that TF1 will call
    def fit_model(x: np.ndarray, par: np.ndarray) -> float:
        # First 3 parameters are S, mu, sigma for the signal Gaussian
        signalVal = signal(x, [par[0], par[1], par[2]])
        # Next 6 parameters are for the reflected background
        # par[3] is the reflected/signal ratio, so it must be multiplied by S=par[0] before being passed
        reflVal = reflected_background(x, [par[0]*par[3], par[4], par[5], par[6], par[7], par[8]])
        # Last backgroundNPar parameters are for the combinatorial background
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
