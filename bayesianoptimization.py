from baseoptimization import BaseOptimization
import matplotlib.pyplot as plt
import time
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.plots import plot_convergence, plot_evaluations, plot_objective


class BayesianOptimization(BaseOptimization):
    def __init__(self, sorter, recording, gt_sorting, params_to_opt,
                 space=None, run_schedule=[150, 100],
                 metric='accuracy', recdir=None,
                 outfile='results'):
        BaseOptimization.__init__(self, sorter=sorter, recording=recording,
                                  gt_sorting=gt_sorting,
                                  params_to_opt=params_to_opt,
                                  space=space, run_schedule=run_schedule,
                                  metric=metric, recdir=None, outfile=outfile)
        self.results_obj = None

    def run(self):
        results = self.optimise(
                self.params_to_opt, self.function_wrapper, self.run_schedule)
        best_parameters = {
            key: results.x[i] for i, key in enumerate(self.params_to_opt)
            }
        results['optimal_params'] = best_parameters
        results.specs['args']['func'] = None
        self.save_results(results, self.outfile)
        self.results_obj = results

    def optimise(self, parameter_definitions, function, run_schedule):
        # Parse parameter definitions to a list of skopt dimensions
        dimensions = []
        for (name, rang) in parameter_definitions.items():
            if type(rang[1]) is int and type(rang) is not list:
                dimensions.append(Integer(low=rang[0], high=rang[1], name=name))
            if type(rang[1]) is float and type(rang) is not list:
                dimensions.append(Real(low=rang[0], high=rang[1], name=name))
            if type(rang) is list:
                dimensions.append(Categorical(rang, name=name))
        run_schedule = self.run_schedule
        start_time = time.time()

        results_object = gp_minimize(function,
                                     dimensions,
                                     acq_func='EI',
                                     noise=1e-10,
                                     n_calls=run_schedule[0],
                                     n_random_starts=run_schedule[1],
                                     n_jobs=-1,
                                     verbose=True
                                     )
        print("--- %s seconds ---" % (time.time() - start_time))
        return results_object

    def plot_convergence(self):
        if self.results_obj is not None:
            parameter_names = list(self.params_to_opt.keys())

            plt.figure(figsize=(15, 3.5))
            plot_convergence(self.results_obj, ax=plt.gca())
