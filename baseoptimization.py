import logging
import os
import pickle
from collections import OrderedDict
from contextlib import redirect_stdout
from multiprocessing import Process, Queue, cpu_count
import numpy as np
import psutil
from tqdm import tqdm
import time
import sys
import shutil
import spiketoolkit as st
import spikesorters as ss
import spikecomparison as sc
from spikeforest import SFMdaSortingExtractor
import spikeforest_analysis as sa
from mountaintools import client as mt


class BaseOptimization(object):
    def __init__(self, sorter, recording, gt_sorting, params_to_opt,
                 space=None, run_schedule=[50, 50], metric='accuracy',
                 recdir=None, outfile=None, x0=None, y0=None):
        self.sorter = sorter.lower()
        self.re = recording
        self.gt_se = gt_sorting
        self.params_to_opt = OrderedDict(params_to_opt)
        self.outfile = outfile
        self.run_schedule = run_schedule
        self.space = space
        self.best_parameters = None
        self.iteration = 0
        self.metric = metric.lower()
        self.recdir = recdir
        self.results_obj = None
        self.SorterClass = ss.sorter_dict[self.sorter]
        self.true_units_above = None
        self.x0 = x0 
        self.y0 = y0
        
        if self.metric == 'spikeforest':
            
            tmp_dir = 'test_outputs_spikeforest'
            if not os.path.exists(tmp_dir):
                print('Creating folder {} for temporary data - note this is not cleaned up.'.format(tmp_dir))
                os.makedirs(tmp_dir)
            SFMdaSortingExtractor.write_sorting(sorting=self.gt_se,
                                                save_path=os.path.join(tmp_dir,'firings_true.mda'))
            print('Compute units info...')
            sa.ComputeUnitsInfo.execute(recording_dir=self.recdir,
                                        firings=os.path.join(tmp_dir,'firings_true.mda'),
                                        json_out=os.path.join(tmp_dir,'true_units_info.json'))

            true_units_info = mt.loadObject(path=os.path.join(tmp_dir,'true_units_info.json'))
            true_units_info_by_unit_id = dict()
            snrthresh = 8
            self.true_units_above = [u['unit_id'] for u in true_units_info if u['snr'] > snrthresh]
            print('Only testing ground truth units with snr > 8: ',self.true_units_above)

    def run(self):
        raise(NotImplementedError)

    def optimise(self, parameter_definitions, function, run_schedule):
        raise(NotImplementedError)

    def save_results(self, file_name):
        with open("{}.pickle".format(file_name), 'wb+') as f:
            pickle.dump([self.results_obj, self.params_to_opt, self.iteration], f)

    def load_results(self, file_name):
        with open("{}.pickle".format(file_name), 'rb') as f:
            self.results_obj, self.params_to_opt, self.iteration = pickle.load(f)

    def delete_folder(self, folder_name):
        shutil.rmtree(str(folder_name), ignore_errors=True)

    def function_wrapper(self, chosen_values):

        print("Iteration {}".format(self.iteration))
        print("Clustering with parameters: {}".format(chosen_values))
        print('', end='', flush=True)

        chosen_parameters = {}
        for i, key in enumerate(self.params_to_opt):
            if type(chosen_values) is dict:
                chosen_parameters[key] = chosen_values[key]
            else:
                chosen_parameters[key] = chosen_values[i]
        output_folder = 'optimization_{}'.format(self.iteration)
        self.iteration += 1

#         logging.info("clustering spikes with parameters: {}".format(chosen_values))

        sorter_par = self.SorterClass.default_params()
        try:
            sorter = self.SorterClass(recording=self.re, output_folder=output_folder)
            sorter_params = sorter.default_params()
            for key in chosen_parameters:
                if key in sorter_par.keys():
                    if type(sorter_par[key]) == int:
                        sorter_par[key] = int(chosen_parameters[key])
                    else:
                        sorter_par[key] = chosen_parameters[key]
            if self.sorter == 'herdingspikes':
                sorter_par['filter'] = False
                sorter_par['pre_scale'] = False
            print("Passed parameters: {}".format(sorter_par))
            print('', end='', flush=True)
            sorter.set_params(**sorter_par)
            sorter.run()
        except Exception as e:
            del sorter
            self.delete_folder(output_folder)
            print('Sorter failed for these parameters. Output:')
            print(e)
            return 1e-4

        sorting_extractor = sorter.get_result()
        if len(sorting_extractor.get_unit_ids()) > 0:
            score = self.compute_score(sorting_extractor)
        else:
            print('Sorter found no units')
            score = 1e-4

        del sorting_extractor
        del sorter
        self.delete_folder(output_folder)
        
        print('score: ', score)
        return score

    def compute_score(self, sorting_extractor):
        if self.metric != 'spikeforest':
            comparison = sc.compare_sorter_to_ground_truth(self.gt_se,
                                                              sorting_extractor, exhaustive_gt=True)
            d_results = comparison.get_performance(method='pooled_with_average', output='dict')
            print('results')
            print(d_results)
            if self.metric == 'accuracy':
                score = d_results['accuracy']
            if self.metric == 'precision':
                score = d_results['precision']
            if self.metric == 'recall':
                score = d_results['recall']
            if self.metric == 'f1':
                print('comparison:')
                print(d_results)
                if (d_results['precision']+d_results['recall']) > 0:
                    score = 2 * d_results['precision'] * d_results['recall'] / (d_results['precision']+d_results['recall'])
                else:
                    score = 0
            del comparison
        else:
            tmp_dir = 'test_outputs_spikeforest'
            SFMdaSortingExtractor.write_sorting(sorting=sorting_extractor, save_path=os.path.join(tmp_dir,'firings.mda'))
            print('Compare with ground truth...')
            sa.GenSortingComparisonTable.execute(firings=os.path.join(tmp_dir,'firings.mda'),
                                                 firings_true=os.path.join(tmp_dir,'firings_true.mda'),
                                                 units_true=self.true_units_above,  # use all units
                                                 json_out=os.path.join(tmp_dir,'comparison.json'),
                                                 html_out=os.path.join(tmp_dir,'comparison.html'),
                                                 _container=None)
            comparison = mt.loadObject(path=os.path.join(tmp_dir,'comparison.json'))
            score = np.mean([float(u['accuracy']) for u in comparison.values()])
        return -score

    def plot_convergence(self):
        raise(NotImplementedError)
