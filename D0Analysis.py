import ROOT as r
from enum import Enum, auto
import hist
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def run_by_run_num_candidates(dataDir, mcDir, runListFull, runList1, runList2, mmin=1.81, mmax=1.90, text=False, useLumi=True):
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
        lumiFile = r.TFile.Open("~/cernbox/PbPb23_singlegap/LHC23_PbPb_pass4_train355821/mergedAnalysisResults.root")
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
    fig = plt.figure(figsize=(20, 5))
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
    ax.set_ylabel("Fraction of run total")
    patchData = mpatches.Patch(color=dictColors['data2'], label='Data' + (' lumi' if useLumi else ''))
    patchMc = mpatches.Patch(color=dictColors['mc2'], label='Mc')
    ax.legend(handles=[patchData, patchMc])
    plt.tight_layout()
    plt.show()

    return ax, listDataNumCandidates, listMcNumCandidates

def run_by_run_efficiency(numeratorString, denominatorString, numeratorTitle, denominatorTitle, numeratorDir, denominatorDir, runListFull, runList1, runList2, minPt=0., maxPt=20., text=False):
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
    meanEffRunList1 = totalNumCountsRunList1 / totalDenCountsRunList1
    meanEffErrRunList1 = (1/totalDenCountsRunList1) * np.sqrt(totalNumCountsRunList1 * (1 - totalNumCountsRunList1/totalDenCountsRunList1)) # Binomial error calculation

    totalNumErrRunList2 = np.sqrt(totalNumCountsRunList2)
    totalDenErrRunList2 = np.sqrt(totalDenCountsRunList2)
    meanEffRunList2 = totalNumCountsRunList2 / totalDenCountsRunList2
    meanEffErrRunList2 = (1/totalDenCountsRunList2) * np.sqrt(totalNumCountsRunList2 * (1 - totalNumCountsRunList2/totalDenCountsRunList2)) # Binomial error calculation
        
    runAxis = np.arange(len(runListFull))
    plt.figure(figsize=(15, 5))
    plt.title(f"$\\frac{{\\text{{{numeratorTitle}}}}}{{\\text{{{denominatorTitle}}}}}$, {minPt} < $p_\\text{{T}}$ < {maxPt} GeV/c")
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
        with open('/home/sigurd/cernbox/PbPb23_singlegap/LHC23_PbPb_pass4_train355821/runInfo.txt', mode='r') as file:
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

    def replace_displayed_title(self, includeshort=False):
        # Replace the displayed title of the histogram with the TeX formatted title
        self.histogram.SetTitle("")
        self.draw_title(0.5, 0.95, size=0.035, includeshort=includeshort)

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

    DEFAULT_RUNLIST = ['545367', '545345', '545332', '545312', '545311', '545296', '545294', '545291', '545289', '545262', '545249', '545246', '545223', '545222', '545210', '545185', '545184', '545171', '545117', '545103', '545086', '545064', '545063', '545062', '545060', '545047', '545044', '545042', '545041', '545009', '545008', '545004', '544992', '544991', '544968', '544964', '544963', '544961', '544947', '544931', '544917', '544914', '544913', '544896', '544887', '544886', '544868', '544813', '544797', '544794', '544767', '544754', '544742', '544739', '544696', '544694', '544693', '544692', '544674', '544672', '544653', '544652', '544640', '544614', '544585', '544583', '544582', '544580', '544568', '544567', '544565', '544564', '544551', '544550', '544549', '544548', '544518', '544515', '544514', '544512', '544511', '544510', '544508', '544492', '544491', '544490', '544477', '544476', '544475', '544474', '544454', '544392', '544391', '544390', '544389', '544185', '544184', '544124', '544123', '544122', '544116', '544098', '544032', '544028', '544013']
    class FileStructures(Enum):
        RUNDIRS = auto()
        FLAT = auto()

    def __init__(self, pathTableMaker, dirData, dirRec, dirGen, ptBins = [0., 0.75, 1.5, 2.25, 3., 4., 6., 8., 12.], runList = None, **kwargs):
        # The runList defines which files are looped over in run-by-run calculations
        self.runList = runList if runList is not None else Analysis.DEFAULT_RUNLIST
        self.runList = sorted(self.runList)
        print(f"This analysis contains {len(self.runList)} runs")

        # Import table-maker merged output file and check that it contains the correct runs
        self.fileTableMaker = r.TFile.Open(pathTableMaker)
        self.check_file_for_runs(self.fileTableMaker, ['bc-selection-task', 'hCounterTCE'])

        self.dirData = dirData
        self.dirRec = dirRec
        self.dirGen = dirGen

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
        # pT bins to be used in the differential cross section
        self.ptBinsArray = ptBins
        self.ptBins = []
        for i in range(len(self.ptBinsArray) - 1):
            self.ptBins.append(PtBin(i, self.ptBinsArray[i], self.ptBinsArray[i+1]))
        # Range of rapidity bin to be used
        self.minY = -0.9
        self.maxY = 0.9
        # Get the mass histograms needed for the signal extraction
        self.prepare_histograms()
        # Get the integrated luminosity from the bc-selection-task
        self.lumi = self.histLumi.Integral() # 1/µb
        print(f"Integrated luminosity (TCE trigger, after BC cuts): {self.lumi} 1/µb")
        self.runByRunLumi = {}
        for run in self.runList:
           self.runByRunLumi[run] = self.histLumi.GetBinContent(self.histLumi.GetXaxis().FindBin(run)) 
        # Set the range of the mass axis for plotting and integrating reflected background
        self.massRange = [0., 4.]
        self.reweighting = False

    def check_file_for_runs(self, file, strings):
        if len(strings) == 3:
            group, subGroup, histName = strings
            hist = file.Get(group).FindObject(subGroup).FindObject(histName)
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
            uniqueLabels = set(self.runList) - set(labels)
            print(f"Items unique to the runList: {uniqueRuns}")
            print(f"Items unique to the file: {uniqueLabels}")
            raise Exception(f"File '{file.GetName()}' does not contain the correct set of runs!")

    def prepare_histograms(self):
        self.groupNameD0Generated = "MCTruthGenAfterBcCuts_D0FS"
        self.groupNameD0PtMatched = f"PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4_{self.pairCutName}_KPiFromD0FS"

        self.histLumi = self.fileTableMaker.Get("bc-selection-task").Get("hLumiTCEafterBCcuts")

        # Obtain the main gen. lvl. histogram used for efficiency, and also all the histograms used for factorized efficiencies later
        fullNamesGen = ["MCTruthGenRec_D0FS", "MCTruthGenSel_D0FS", "MCTruthGenSelDaughtersInAcc_KPiFromD0FS", "MCTruthGenRecDaughtersInAcc_KPiFromD0FS", "MCTruthGenAfterBcCutsDaughtersInAcc_KPiFromD0FS"]
        fullNamesGen = ["analysis-asymmetric-pairing/output;1/" + name + "/PtMC_YMC" for name in fullNamesGen]
        fullNameHistD0PtYGenerated = "analysis-asymmetric-pairing/output;1/" + self.groupNameD0Generated + "/MyMcPtYHisto"
        fullNamesGen.append(fullNameHistD0PtYGenerated)
        self.dictGenHists = self.get_histograms(self.dirGen, fullNamesGen)
        self.histD0PtYGenerated = self.dictGenHists[fullNameHistD0PtYGenerated]
        # Project out our dy bin from the gen histogram
        lowerYBin = self.histD0PtYGenerated.GetYaxis().FindBin(self.minY)
        upperYBin = self.histD0PtYGenerated.GetYaxis().FindBin(self.maxY) - 1
        self.histD0PtGenerated = self.histD0PtYGenerated.ProjectionX(f"projPtMcGen_y_{self.minY}_{self.maxY}", lowerYBin, upperYBin)
        self.histD0PtGenerated.SetTitle(f"Generated D0 after BC cuts in {self.minY} < y < {self.maxY}")
        # Data histograms
        fullNameHistD0MassPt = "analysis-asymmetric-pairing/output;1/" + f"PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_{self.pairCutName}" + "/MyMassPtHisto"
        fullNameHistEventAfterCuts = "analysis-event-selection/output;1/Event_AfterCuts/VtxZ"
        tmpDict = self.get_histograms(self.dirData, [fullNameHistD0MassPt, fullNameHistEventAfterCuts])
        self.histD0MassPt = tmpDict[fullNameHistD0MassPt]
        self.histEventAfterCuts = tmpDict[fullNameHistEventAfterCuts]
        del tmpDict
        # Obtain the main rec. matched histogram, the reflected histogram, and the rec. lvl. histograms used for factorized efficiencies later
        fullNamesRec = ["noTrackCut:noTrackCut", f"{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4", f"{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4_{self.pairCutName}"]
        fullNamesRec = ["analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_" + fullNameRec + "_KPiFromD0FS/Pt" for fullNameRec in fullNamesRec]
        fullNameHistD0PtMatched = "analysis-asymmetric-pairing/output;1/" + self.groupNameD0PtMatched + "/Pt"
        fullNamesRec.append(fullNameHistD0PtMatched)
        fullNameHistD0MassPtReflected = "analysis-asymmetric-pairing/output;1/" + f"{self.groupNameD0PtMatched}Reflected" + "/MyMassPtHisto"
        fullNamesRec.append(fullNameHistD0MassPtReflected)
        self.dictRecHists = self.get_histograms(self.dirRec, fullNamesRec)
        self.histD0PtMatched = self.dictRecHists[fullNameHistD0PtMatched]
        self.histD0PtMatched.SetName("histD0PtMatched")
        self.histD0PtMatched.SetTitle("Reconstructed, matched D0")
        self.histD0MassPtReflected = self.dictRecHists[fullNameHistD0MassPtReflected]

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

    def get_histograms(self, directory, fullHistogramNames, histogramNamesfileStructure = None):
        if histogramNamesfileStructure is None:
            histogramNamesfileStructure = self.FileStructures.FLAT
        splitHistNames = [s.split("/") for s in fullHistogramNames] 
        namesDict = {}
        for irow, row in enumerate(splitHistNames):
            depth = len(row)
            if row[1] == 'output;1':
                depth -= 1
                row[0] += ('/' + row[1])
                row = np.delete(row, 1)
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

            else:
                raise Exception(f"Histogram name with depth = {depth}, don't know what to do!")

        histograms = {}
        print(f"Getting histograms from {directory}...")
        if depth == 3:
            for irun, run in enumerate(self.runList):
                print(f"Processing run {run} ({irun+1}/{len(self.runList)})...            ", end='\r')
                if (histogramNamesfileStructure == self.FileStructures.FLAT):
                    filePath = f"{directory}/AnalysisResults_run{run}.root"
                elif (histogramNamesfileStructure == self.FileStructures.RUNDIRS):
                    filePath = f"{directory}/{run}/AnalysisResults.root"
                else:
                    raise Exception("Not a valid FileStructure!")
                with uproot.open(filePath) as file:
                    for dirName in namesDict.keys():
                        dir = file[dirName]
                        for i, item in enumerate(dir):
                            if item.member("fName") in namesDict[dirName]:
                                for ii, iitem in enumerate(dir[i]):
                                    if iitem.member("fName") in namesDict[dirName][item.member("fName")]:
                                        fullName = dirName + '/' + item.member("fName") + '/' + iitem.member("fName")
                                        tmpHist = dir[i][ii]
                                        tmpHistPr = tmpHist.to_pyroot()
                                        if fullName not in histograms:
                                            histograms[fullName] = tmpHistPr
                                        else:
                                            histograms[fullName].Add(tmpHistPr)
        elif depth == 2:
            for irun, run in enumerate(self.runList):
                print(f"Processing run {run} ({irun+1}/{len(self.runList)})...            ", end='\r')
                if (histogramNamesfileStructure == self.FileStructures.FLAT):
                    filePath = f"{directory}/AnalysisResults_run{run}.root"
                elif (histogramNamesfileStructure == self.FileStructures.RUNDIRS):
                    filePath = f"{directory}/{run}/AnalysisResults.root"
                else:
                    raise Exception("Not a valid FileStructure!")
                with uproot.open(filePath) as file:
                    for dirName in namesDict.keys():
                        dir = file[dirName]
                        for i, item in enumerate(dir):
                            if item.member("fName") in namesDict[dirName]:
                                fullName = dirName + '/' + item.member("fName")
                                tmpHist = dir[i]
                                tmpHistPr = tmpHist.to_pyroot()
                                if fullName not in histograms:
                                    histograms[fullName] = tmpHistPr
                                else:
                                    histograms[fullName].Add(tmpHistPr)

        print("\nDone!")
        return histograms

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
        else:
            raise Exception(f"Invalid background function '{backgroundName}'")

        # Fit the histogram
        print("---- Fitting data ----")
        self.ptBins[bin].fitResult = self.ptBins[bin].massPtSlice.Fit(fitFunc, "L0S", "", fitRange[0], fitRange[1])

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
            recHist = recHistRaw.to_pyroot()
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
            genHistPtY = genHistRaw.to_pyroot()

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
        self.efficiencyReweighted.SetTitle("Reconstructed / Generated, reweighted")
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
        self.efficiencyWithoutReweighting.SetTitle("Reconstructed / Generated, without reweighting")
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
        fullNames = ["analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_" + fullName + "_KPiFromD0FS/Pt" for fullName in fullNames]
        tmpDict = self.get_histograms(directory, fullNames)
        # Reconstructed, matched D0 in selected events, but with no track or pair cuts. This is the denominator for all partial efficiencies due to track cuts
        histDenom = tmpDict["analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_KPiFromD0FS/Pt"]
        histDenom = histDenom.Rebin(len(self.ptBinsArray) - 1, histDenom.GetName(), np.asarray(self.ptBinsArray, 'd'))

        histD0PtKaonTPC = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{trackCutNames['kaonTPCCut']}:noTrackCut_KPiFromD0FS/Pt"]
        histD0PtKaonTPC = histD0PtKaonTPC.Rebin(len(self.ptBinsArray) - 1, histD0PtKaonTPC.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., kaon TPC nSigma<3", "Rec. matched D0 in sel evt.", histD0PtKaonTPC, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtKaonTOF = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{trackCutNames['kaonTOFCut']}:noTrackCut_KPiFromD0FS/Pt"]
        histD0PtKaonTOF = histD0PtKaonTOF.Rebin(len(self.ptBinsArray) - 1, histD0PtKaonTOF.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., kaon TOF nSigma<3", "Rec. matched D0 in sel evt.", histD0PtKaonTOF, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtEtaCut = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['etaCut']}_KPiFromD0FS/Pt"]
        histD0PtEtaCut = histD0PtEtaCut.Rebin(len(self.ptBinsArray) - 1, histD0PtEtaCut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track |eta|<0.9", "Rec. matched D0 in sel evt.", histD0PtEtaCut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtPtCut = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['ptCut']}_KPiFromD0FS/Pt"]
        histD0PtPtCut = histD0PtPtCut.Rebin(len(self.ptBinsArray) - 1, histD0PtPtCut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track pT>0.5 GeV/c", "Rec. matched D0 in sel evt.", histD0PtPtCut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtITSibCut = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['itsQualityCut']}_KPiFromD0FS/Pt"]
        histD0PtITSibCut = histD0PtITSibCut.Rebin(len(self.ptBinsArray) - 1, histD0PtITSibCut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track IsITSibAny=1", "Rec. matched D0 in sel evt.", histD0PtITSibCut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtTPCnclsCut = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['tpcNClsCut']}_KPiFromD0FS/Pt"]
        histD0PtTPCnclsCut = histD0PtTPCnclsCut.Rebin(len(self.ptBinsArray) - 1, histD0PtTPCnclsCut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track TPC nCls > 50", "Rec. matched D0 in sel evt.", histD0PtTPCnclsCut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtDCAzCut = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['dcaZCut']}_KPiFromD0FS/Pt"]
        histD0PtDCAzCut = histD0PtDCAzCut.Rebin(len(self.ptBinsArray) - 1, histD0PtDCAzCut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track |DCAz| < 0.3 cm", "Rec. matched D0 in sel evt.", histD0PtDCAzCut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtTPCchi2Cut = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_{trackCutNames['tpcChi2Cut']}_KPiFromD0FS/Pt"]
        histD0PtTPCchi2Cut = histD0PtTPCchi2Cut.Rebin(len(self.ptBinsArray) - 1, histD0PtTPCchi2Cut.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., track TPCchi2 < 4", "Rec. matched D0 in sel evt.", histD0PtTPCchi2Cut, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff
        histD0PtAllTrackCuts = tmpDict[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{trackCutNames['kaonFullCut']}:noTrackCut_{trackCutNames['fullCommonCut']}_KPiFromD0FS/Pt"]
        histD0PtAllTrackCuts = histD0PtAllTrackCuts.Rebin(len(self.ptBinsArray) - 1, histD0PtAllTrackCuts.GetName(), np.asarray(self.ptBinsArray, 'd'))
        eff = Efficiency("Rec. matched D0 in sel evt., after all track cuts", "Rec. matched D0 in sel evt.", histD0PtAllTrackCuts, histDenom)
        self.trackCutEfficiencies.append(eff)
        del eff

    def calculate_factorized_efficiencies(self):
        """
        Calculate factorized efficiencies for some predefined (hardcoded) factorizations
        Total efficiency = N(rec. matched D0 after all cuts) / N(gen. D0 after BC cuts)
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

        self.histD0PtMatchedInSelEvent = self.dictRecHists[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_noTrackCut:noTrackCut_KPiFromD0FS/Pt"]
        self.histD0PtMatchedInSelEvent = self.histD0PtMatchedInSelEvent.Rebin(len(self.ptBinsArray) - 1, f"PtMcMatchedInSelEvent", np.asarray(self.ptBinsArray, 'd'))
        self.histD0PtMatchedInSelEvent.SetTitle("Reconstructed, matched D0->Kpi in selected event, no track or pair cuts")

        self.histD0PtMatchedInSelEventAfterTrackCuts = self.dictRecHists[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4_KPiFromD0FS/Pt"]
        self.histD0PtMatchedInSelEventAfterTrackCuts = self.histD0PtMatchedInSelEventAfterTrackCuts.Rebin(len(self.ptBinsArray) - 1, f"PtMcMatchedInSelEventAfterTrackCuts", np.asarray(self.ptBinsArray, 'd'))
        self.histD0PtMatchedInSelEventAfterTrackCuts.SetTitle("Reconstructed, matched D0->Kpi in selected event, selected tracks, no pair cuts")

        if not hasattr(self, 'histD0PtMatchedFinalBins'):
            self.histD0PtMatchedInSelEventAfterTrackCutsAndPairCuts = self.dictRecHists[f"analysis-asymmetric-pairing/output;1/PairsBarrelSEPM_{self.kaonLegCutName}:{self.pionLegCutName}_singleGapTrackCuts4_{self.pairCutName}_KPiFromD0FS/Pt"]
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

    def create_raw_yield_histogram(self):
        self.histRawYield = r.TH1F("histRawYield", "Raw yield /#Delta p_{T}, raw stat. errors", len(self.ptBins), np.asarray(self.ptBinsArray, 'd'))
        self.histRawYield.GetYaxis().SetTitle("Raw D^{0} yield (1/GeV c^{-1})")
        self.histRawYield.GetXaxis().SetTitle("p_{T} (GeV c^{-1})")
        for i, bin in enumerate(self.ptBins):
            self.histRawYield.SetBinContent(i+1, bin.nSignal / self.histRawYield.GetBinWidth(i+1))
            # Error propagation with the bin width
            self.histRawYield.SetBinError(i+1, bin.relativeStatError / self.histRawYield.GetBinWidth(i+1) * bin.nSignal)
        self.histRawYield.SetStats(0)

    def calculate_spectrum_per_event(self):
        self.nEvents = self.histEventAfterCuts.GetEntries()
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

    def calculate_raw_yield_per_lumi(self, draw=True):
        self.histRawYieldPerLumi = self.histRawYield.Clone()
        self.histRawYieldPerLumi.SetName("histRawYieldPerLumi")
        self.histRawYieldPerLumi.SetTitle("Raw yield /#Delta p_{T} L_{int}")
        self.histRawYieldPerLumi.Scale(1 / self.lumi) # µb / GeVc^-1
        self.histRawYieldPerLumi.Scale(1 / 1000.) # mb / GeVc^-1
        self.histRawYieldPerLumi.GetYaxis().SetTitle("Raw D^{0} yield / L_{int} (mb/GeV c^{-1})")
        self.histRawYieldPerLumi.SetStats(0)
        if draw:
            self.canvasRawYieldPerLumi = r.TCanvas("canvasRawYieldPerLumi")
            self.canvasRawYieldPerLumi.cd()
            self.histRawYieldPerLumi.Draw()
            r.gPad.SetLogy()
            self.canvasRawYieldPerLumi.Draw()

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


    def draw_fits_and_yield(self):
        # Figure out grid layout
        nPanels = len(self.ptBins) + 1
        nRows = int(np.ceil(nPanels/3))
        self.canvasFitsYields = r.TCanvas("canvasFitsYields", "canvasFitsYields", 1200, nRows * 333)
        self.canvasFitsYields.Divide(3, nRows, 0.002, 0.01)

        self.textBoxesFitsYields = []
        self.linesFitrangeLow = []
        self.linesFitrangeHigh = []
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

        for i, bin in enumerate(self.ptBins):
            self.canvasFitsYields.cd(i+1)
            # Lines to show fitting range
            self.linesFitrangeLow.append(r.TLine(bin.fitRange[0], r.gPad.GetUymin(), bin.fitRange[0], r.gPad.GetUymax()))
            self.linesFitrangeLow[i].SetLineColor(r.kBlack)
            self.linesFitrangeLow[i].SetLineStyle(2)
            self.linesFitrangeLow[i].SetLineWidth(1)
            self.linesFitrangeLow[i].Draw()
            self.linesFitrangeHigh.append(r.TLine(bin.fitRange[1], r.gPad.GetUymin(), bin.fitRange[1], r.gPad.GetUymax()))
            self.linesFitrangeHigh[i].SetLineColor(r.kBlack)
            self.linesFitrangeHigh[i].SetLineStyle(2)
            self.linesFitrangeHigh[i].SetLineWidth(1)
            self.linesFitrangeHigh[i].Draw()

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
