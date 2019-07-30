from baseoptimization import BaseOptimization
# import spiketoolkit as st
# import spikeextractors as se
import numpy as np
# import logging
# import pickle
# from contextlib import redirect_stdout
# from multiprocessing import Process, Queue, cpu_count

# import psutil
import matplotlib.pyplot as plt
# from tqdm import tqdm
import time
# import os, sys
from collections import defaultdict

import hyperopt
from hyperopt.pyll.stochastic import sample
from hyperopt import Trials
from hyperopt import tpe
from hyperopt import hp



class TPEOptimization(BaseOptimization):
    def __init__(self,sorter,recording,gt_sorting, params_to_opt,space,run_schedule=[100],metric ='accuracy',recdir=None,
                 outfile='results'):
        BaseOptimization.__init__(self, sorter=sorter, recording=recording, gt_sorting=gt_sorting, params_to_opt=params_to_opt, space=space, run_schedule=run_schedule, metric=metric, recdir = None, outfile=outfile)
        self.trials = Trials()
            
    def run(self):
        results = self.optimise(
                self.params_to_opt, self.function_wrapper, self.run_schedule)
        self.save_results(results, self.outfile)
            
    def optimise(self, parameter_definitions, function, run_schedule):
        start_time = time.time()
        best = hyperopt.fmin(function, self.space, algo=tpe.suggest, max_evals = run_schedule[0], trials=self.trials)
        results_obj = defaultdict()
        results_obj['optimal_params']=self.trials.best_trial['misc']['vals']
        results_obj['best_accuracy']=-self.trials.best_trial['result']['loss']
        print("--- %s seconds ---" % (time.time() - start_time))
        return results_obj
        
    def plot_convergence(self):
        
        ys = [t['result']['loss'] for t in self.trials.trials]
        
        plt.figure(figsize=(15,3.5))
        ax = plt.gca()
        ax.grid()
        n_calls = len(ys)
        mins = [np.min(ys[:i]) for i in range(1, n_calls + 1)]
        ax.plot(range(1, n_calls + 1), mins, c='b', marker=".", markersize=12, lw=2)
        plt.xlabel('n_calls')
        plt.ylabel('min(-Accuracy)')
        plt.title('Convergence of TPE in sorting optimisation')
        
    def plot_histograms(self):
        
        parameters = list(trials.trials[0]['misc']['vals'].keys())
        n = len(parameters)
        cmap = plt.cm.jet
        for i, val in enumerate(parameters):
            xs = np.array([t['misc']['vals'][val] for t in self.trials.trials]).ravel()
            ys = [-t['result']['loss'] for t in trials.trials]
  
            ys = np.array(ys)
            plt.figure(figsize=(3,3))
            plt.hist(xs)
            plt.title(val)

