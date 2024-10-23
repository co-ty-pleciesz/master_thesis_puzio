import os, sys
import matplotlib.pyplot as plt
from mne import make_fixed_length_epochs
from toolkit import *
import torch
import time
import glob
import multiprocessing

num_cpus = multiprocessing.cpu_count()
print(f'Liczba dostepnych cpu: {num_cpus}')
start_time = time.time()
setCwdHere()
loadIDTxl()

from idtxl.multivariate_te import MultivariateTE    # IDTxl: multivariate transfer entropy class
from idtxl.visualise_graph import plot_network      # IDTxl: plotting class

def worker(num):
    '''Funckcja wykonujaca caly program'''

    device = torch.device("cpu")

    # Set the device for IDTxl
    network_analysis = MultivariateTE()
    network_analysis.set_device(device)

    # Open the new log file for writing
    sys.stdout = open(f"logs_multivariate.txt", "w")

    # Rest of the code...
    srcDir = '/home/syl/Documents/master/sylw_pociete/output_art'
    subCode = 'ARZ000'
    cond = 'art_watch2'
    samplingRate = 1000 # ?    potestowac
    samplesPerMs = samplingRate / 1000
    eeg = loadRawEEG_epochs(srcDir, subCode, cond)

    data = adjustSignalToIDTxl(eeg, containesEpochedData=True)

    # setup TE analysis
    minLagInMs = 1
    maxLagInMs = 50  # ile ms wstecz sprawdzac`

    settings = {
        'cmi_estimator': 'JidtGaussianCMI',
        'n_perm_max_stat': 100,
        'n_perm_min_stat': 100,
        'n_perm_omnibus': 100,
        'n_perm_max_seq': 100,
        'max_lag_sources': int(maxLagInMs * samplesPerMs),
        'min_lag_sources': int(minLagInMs * samplesPerMs),
        "alpha_min_stat": 0.05,
        "alpha_max_stat": 0.05,
        "alpha_omnibus": 0.05,
        "alpha_max_seq": 0.05,
    }

    # Moving multivariate TE analysis
    moving_te_settings = {
        "timeRange": [0, data.data.shape[1] - 1],
        "pastSpan": 100,#50,  #0 lub all, ale wszystkie; number of samples to look into the past
        "step": 50,  #1,  50 - 100
        "targets": [2], #,],   czy dobrze robie i oba sa target i source?     #ustaw tu for ze jedna liczba jest targetem, a reszta sourcem i potem daleej w forze, ze kolejny targetem jest kolejnym i reszta sorcem
        #funkcja set do tego 
    #ile czasu 
        "sources": [1],#,3,4,5,6,7,12,13,14,15,16,23,24,25,26,27,29,31],  # list of sources
        "cmi_estimator": "JidtGaussianCMI", #box-kernel - biased, kraskov- slower, best
        "fdr_correction": True,
    }
    #1. multivariate -> bivariate bez zmiany parametrow - ile sie liczy dla 20, ile dla 1sorce'a
    #2. pastSpan": 100,
    #  "step": 50, 
    #3. czy dane moge wrzucić - 1 czlwieka dane ew
    #statystyczne moge 
    #fdr correction :false moge 
    #sroda 16.15 ->discord


    # Run moving TE analysis
    resultList, aux = computeMovingMultivariateTransferEntropy(data, moving_te_settings)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Czas wykonania programu(bez wyswietlenia matrixa): {execution_time} sekund")

    # Create TE matrix for the last window
    num_targets = len(moving_te_settings["targets"])
    num_sources = len(moving_te_settings["sources"])
    te_matrix = np.zeros((num_sources, num_targets))
    last_result = resultList[-1]


    for target in moving_te_settings["targets"]:
        plotSingleTargetMteTimeSeries(resultList, target)


    for target in moving_te_settings["targets"]:
        
        mTE_vs_source_time = createSourceTargetMteDict(resultList, aux, target=target)

        for source in moving_te_settings["sources"]:
            plotSigleSourceTargetMteTimeSeries(
                mTE_vs_source_time, source=source, targetLabel=str(target)
            )


    # Plot inferred network to console and via matplotlib for the last window
    resultList.print_edge_list(weights='max_te_lag', fdr=False)
    plot_network(results=resultList, weights='max_te_lag', fdr=False)
    plt.show()
    input('Script ended. Press ENTER ...')

processes = []

for i in range(num_cpus):
    process = multiprocessing.Process(target=worker, args=(i,))
    processes.append(process)
    process.start()

for process in processes:
    process.join()

# # Check if GPU is available
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using GPU for computation.")
# else:




# epochLengthMs = 0
# # optionally narrow down epoch selection (for faster testing)
# # epochIndicesList = range(100, 111)
# if epochLengthMs > 0:
#     eeg = make_fixed_length_epochs(eeg, duration=epochLengthMs / 1000, preload=True)  # preload=False ?  
#     data = adjustSignalToIDTxl(eeg, containesEpochedData=True, epochIndices=epochIndicesList)
# else:
#     data = adjustSignalToIDTxl(eeg, containesEpochedData=False)







