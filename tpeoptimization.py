from baseoptimization import BaseOptimization
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import hyperopt
from hyperopt.pyll.stochastic import sample
from hyperopt import Trials
from hyperopt import tpe
from hyperopt import hp


class TPEOptimization(BaseOptimization):
    def __init__(self, sorter, recording, gt_sorting, params_to_opt,
                 space=None, run_schedule=[100], metric='accuracy',
                 recdir=None, outfile='results'):

        BaseOptimization.__init__(self, sorter=sorter, recording=recording,
                                  gt_sorting=gt_sorting,
                                  params_to_opt=params_to_opt,
                                  space=space, run_schedule=run_schedule,
                                  metric=metric, recdir=None, outfile=outfile)
        self.trials = Trials()
        self.space = self.define_space(space)

    def run(self):
        results = self.optimise(
                self.params_to_opt, self.function_wrapper, self.run_schedule)
        self.save_results(results, self.outfile)

    def optimise(self, parameter_definitions, function, run_schedule):

        start_time = time.time()
        best = hyperopt.fmin(function,
                             self.space,
                             algo=tpe.suggest,
                             max_evals=run_schedule[0],
                             trials=self.trials)

        results_obj = self.get_optimization_details()
        results_obj['time_taken'] = start_time - time.time()
        print("--- %s seconds ---" % (time.time() - start_time))
        return results_obj

    def define_space(self, space):
        if space not None:
            return space
        space = {}
        for key, value in self.params_to_opt.items():
            if type(value) is list:
                space[key] = hp.choice(key, value)
            if type(value[0]) is int:
                space[key] = hp.quniform(key, value[0], value[1], 1)
            if type(value[0]) is float:
                space[key] = hp.uniform(key, value[0], value[1])
        return space

    def get_optimization_details(self):
        results_obj = {}
        results_obj['optimal_params'] = self.trials.best_trial['misc']['vals']
        results_obj['best_score'] = -self.trials.best_trial['result']['loss']

        results_obj['params_evaluated'] = self.trials.vals
        results_obj['scores'] = self.results_obj[]
        results_obj['iter_min_found'] = self.trials.best_trial['tid']
        results_obj['trials'] = self.trials
        results_obj['avg_best_score'] = self.trials.average_best_error()
        results_obj['total_iter'] = self.iteration

        return results_obj

    def plot_convergence(self):
        ys = [t['result']['loss'] for t in self.trials.trials]

        plt.figure(figsize=(15, 3.5))
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
            plt.figure(figsize=(3, 3))
            plt.hist(xs)
            plt.title(val)
