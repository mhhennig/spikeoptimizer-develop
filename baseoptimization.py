
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
import os, sys
import shutil
import spiketoolkit as st
from spikeforest import SFMdaSortingExtractor
import spikeforest_analysis as sa




class BaseOptimization(object):
    def __init__(self, sorter, recording, gt_sorting, params_to_opt, space=None, run_schedule=[100, 100], metric = 'accuracy', recdir =None, outfile='results'): 
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
            
    def run(self):
        raise(NotImplementedError)

    def optimise(self, parameter_definitions, function, run_schedule):
        raise(NotImplementedError)

    def save_results(self, obj, file_name):
        with open("{}.pickle".format(file_name), 'wb+') as f:
            pickle.dump(obj, f)
   
    def load_results(self, file_name):
        with open("{}.pickle".format(file_name), 'rb') as f:
#             f.encode() # this throws an error for me, why?
            return pickle.load(f)
        
    def delete_folder(self,folder_name):
        shutil.rmtree(str(folder_name), ignore_errors=True)
        
    def function_wrapper(self, chosen_values):
        chosen_parameters = {}
        print('chosen values:',chosen_values)
        for i, key in enumerate(self.params_to_opt):
            if type(chosen_values) is dict:
                chosen_parameters[key] = chosen_values[key]
            else:
                chosen_parameters[key] = chosen_values[i]
        output_folder = 'optimization_{}'.format(self.iteration)
        self.iteration+=1
                
        print("Clustering spikes with parameters: {}".format(chosen_values))
     
        print('', end='', flush=True)

        logging.info("clustering spikes with parameters: {}".format(chosen_values))

        SorterClass = st.sorters.sorter_dict[self.sorter]
        try:
            sorter = SorterClass(recording=self.re, output_folder = output_folder)
            sorter.set_params(**chosen_parameters)
            sorter.run()
        except:
            print('sorter failed for these parameters')
            return 0

        sorting_extractor = sorter.get_result()
        if len(sorting_extractor.get_unit_ids())>0:
            score = self.compute_score(sorting_extractor)    
        else:
            print('sorter found no units')
            score = 0

        del sorting_extractor
        del sorter

        print('score: ', score)
        return score

    def compute_score(self,sorting_extractor):
        sc = st.comparison.compare_sorter_to_ground_truth(self.gt_se, sorting_extractor, exhaustive_gt=True)
        
        d_results = sc.get_performance(method='pooled_with_sum',output='dict')  
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
                score =2*d_results['precision']*d_results['recall']/(d_results['precision']+d_results['recall'])
            else:
                score = 0
        if self.metric == 'spikeforest':
            sorting_true = self.gt_se
            hs2_se = sorting_extractor
            recdir = self.recdir
            if self.iteration<2:
                SFMdaSortingExtractor.write_sorting(sorting=sorting_true,save_path='test_outputs/firings_true.mda')

            SFMdaSortingExtractor.write_sorting( sorting=hs2_se,save_path='test_outputs/firings.mda')

            # run the comparison
            print('Compare with truth...')
            sa.GenSortingComparisonTable.execute(firings='test_outputs/firings.mda',
                firings_true='test_outputs/firings_true.mda',
                units_true=[],  # use all units
                json_out='test_outputs/comparison.json',
                html_out='test_outputs/comparison.html',
                _container=None
            )

            # we may also want to compute the SNRs of the ground truth units
            # together with firing rates and other information
            print('Compute units info...')
            sa.ComputeUnitsInfo.execute(
                recording_dir=recdir,
                firings='test_outputs/firings_true.mda',
                json_out='test_outputs/true_units_info.json'
            )
            
            # Load and consolidate the outputs
            true_units_info = mt.loadObject(path='test_outputs/true_units_info.json')
            comparison = mt.loadObject(path='test_outputs/comparison.json')
            true_units_info_by_unit_id = dict()
            for unit in true_units_info:
                true_units_info_by_unit_id[unit['unit_id']] = unit
            for unit in comparison.values():
                unit['true_unit_info'] = true_units_info_by_unit_id[unit['unit_id']]
            # Print SNRs and accuracies
            for unit in comparison.values():
                print('Unit {}: SNR={}, accuracy={}'.format(unit['unit_id'], unit['true_unit_info']['snr'], unit['accuracy']))
            # Report number of units found
            snrthresh = 8
            units_above = [unit for unit in comparison.values() if float(unit['true_unit_info']['snr'] > snrthresh)]
            #print('Avg. accuracy for units with snr >= {}: {}'.format(snrthresh, np.mean([float(unit['accuracy']) for unit in units_above])))

            score =  np.mean([float(unit['accuracy']) for unit in units_above])
         
            del hs2_se
       
        return -score

    def plot_convergence(self):
        raise(NotImplementedError)

