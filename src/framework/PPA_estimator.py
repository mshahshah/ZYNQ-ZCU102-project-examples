import os,sys, copy,numpy, yaml, shutil
import argparse, random, glob,pandas, pickle
import os, sys, copy, yaml
import argparse, random, glob, pandas, pickle
import numpy as np
from dnn_estimator import dnn_estimator
from utils import *

class PPA_estimator:
    def __init__(self,cfg, hls_tools, dnn_tools, dnn_analyzer):
        self.cfg = cfg
        self.hls_tools = hls_tools
        self.dnn_tools = dnn_tools
        self.dnn_analyzer = dnn_analyzer
        self.utils = utils(cfg)
        self.ann = []

# ###################################################################
# #############  train model and estimate   #########################

    def build_train_dnn_model(self, cfg, trainner_cfg, dnnEstimator):
        '''
        Generates and trains a DNN model for predicting performance of a design.
        :param cfg:
        :param trainner_cfg:
        :param dnnEstimator:
        :return:
        '''
        method = cfg.design_setting.Modeling['method']
        all_models = {}
        for tr_layer in trainner_cfg['target_training_layer']:
            print('\n'+'-'*100)
            print("----------- Started training {} Layer with {} method ---------- ".format(tr_layer, method))
            dnnEstimator.target_layer = tr_layer
            pickle_label = '{}_{}'.format(tr_layer.split('_')[0].upper(), method)
            dataset_file = os.path.join(dnnEstimator.cfg.paths.ml_dataset, trainner_cfg['dataset'])
            dnn_estimator.rawdata = dnnEstimator.load_raw_dataset(dataset_file, trainner_cfg['shuffle_dataset'])
            models_per_lyr = {}
            for o_label in trainner_cfg['out_labels']:
                dnnEstimator.out_labels = [o_label]
                model, mean, std, best_model_losss, evaluations = dnnEstimator.build_and_evaluate_per_output(method)
                models_per_lyr[o_label] = {'model': model, 'mean': mean, 'std': std, 'method': method, 'eval':evaluations}
                dnnEstimator.save_all_models(models_per_lyr, pickle_label)
            all_models[tr_layer] = models_per_lyr

        for tr_layer in trainner_cfg['target_training_layer']:
            print("\nFor estimation model {}, layer {}: ".format(dnnEstimator.modeling_methods,tr_layer))
            for o_label in trainner_cfg['out_labels']:
                if all_models[tr_layer][o_label]['eval'] is not None:
                    print("Evaluation for {:<8}:{}".format(o_label, self.utils.print_dict(all_models[tr_layer][o_label]['eval'], 8, 20, 5, ' ')))
                else:
                    print("Evaluation for {:<8} is not exist!".format(o_label))
        dnnEstimator.out_labels = trainner_cfg['out_labels']
        return dnnEstimator

    def estimate_dnn_model(self, cfg, dnnEstimator, given_lyrs):
        '''
        Uses dnn_estimator to quickly generate an estimate for design performance.
        :param dnnEstimator: estimator object
        :param given_lyrs:
        :return:
        '''
        log = cfg.design_setting.Modeling.get('run_estimation_log',0)
        start_time = self.utils.record_time()
        layers_estimation = dnnEstimator.estimate_each_layer_param(given_lyrs, log=log)
        print('\nEST: The Estimation for  {} is as below:'.format(cfg.design_setting.topmodule))
        topmodule_estimation = dnnEstimator.estimate_top_module_param(layers_estimation, log=log, norm_type='abs')
        LogicSyn_estimation = dnnEstimator.estimate_LogicSyn_param(topmodule_estimation, log=log, norm_type='zero')
        power_estimation = dnnEstimator.estimate_power(LogicSyn_estimation, log=log, norm_type='zero')
        self.utils.end_and_print_time_and_label(start_time, 'Estimation finished')
        return LogicSyn_estimation, topmodule_estimation, layers_estimation

    def train_model_and_estimate(self, cfg, given_lyrs):
        #if cfg.design_setting.Modeling['DNN_ML_model'] == 'none':
        #    return ''
        estimator_cfg = {
            'random_seed': 1,
            'log_interval': 100,
            'drop_outrange': 0,
            'attenuation_per': 100,
            'clip_per': 100,
            'shuffle_dataset': True,
            'dataset': 'dse_cfg_minimal_zynq100_150M/compacked_599solutions.pickle',
            'save_model_as': 'v2',
            'target_training_layer': ['conv_3DT1_L1', 'ds_3DT1_L2', 'fc_T1_L3'], #read_input3D_L1, conv_3DT1_L1, ds_3DT1_L2, fc_T1_L3
            #'target_training_layer': ['conv_3DT1_L1'],
            'estimation_model': 'skl_GBR',
            'normalize_output': True,
            'plot_log_scale': True,
            'in_labels': ['w_in', 'w_out', 'lyr_in', 'lyr_out', 'w_ker', 'stride'],
            #'out_labels': ['DSP', 'LUT', 'FF']
            'out_labels': ['exec us', 'latency', 'BRAM', 'DSP', 'LUT', 'FF']
            #'out_labels': ['P_Slice', 'P_Block', 'P_DSPs', 'P_Static', 'P_Total']
            #'out_labels': ['Timing_PS']
            #'out_labels': ['exec us']
            #'out_labels': ['power_mw']
        }
        ML_cfg = {
            'MLP_dim': [35, 15, 10],
            'n_epochs': 1000,
            'batchS': 60,
            'lr_rate': 0.005,
            'dataset_ratio': 80,
            'target_loss': 0.00001,
            'log_interval': 1,
            'shuffle_training': True
        }
        log = cfg.design_setting.Modeling['run_estimation_log']
        if cfg.design_setting.Modeling['method'] == 'all':
            methods = ['torch_MLP', 'skl_MLP', 'skl_RFR', 'skl_GBR']
        else:
            methods = [cfg.design_setting.Modeling['method']]

        if not cfg.design_setting.Modeling['DNN_ML_model'] == 'none':
            for method in methods:
                cfg.design_setting.Modeling['method'] = method
                dnnEstimator = dnn_estimator(cfg)
                dnnEstimator.set_trainner_setting(estimator_cfg, ML_cfg)
                dnnEstimator = self.build_train_dnn_model(cfg, estimator_cfg, dnnEstimator)

        if cfg.design_setting.Modeling['run_estimation']:
            dnnEstimator = dnn_estimator(cfg)
            dnnEstimator.set_trainner_setting(estimator_cfg, ML_cfg)
            dnnEstimator.load_trainned_models(log=log)
            LogicSyn_estimation, topmodule_estimation, layers_estimation = self.estimate_dnn_model(cfg, dnnEstimator, given_lyrs)
        else:
            dnnEstimator = dnn_estimator(cfg)
            dnnEstimator.top_estimation = {'exec us': 0, 'latency': 0, 'BRAM': 0, 'DSP': 0, 'LUT': 0,
                                           'FF': 0, 'GOPS': 0}
        return dnnEstimator

    def evaluate_estimation_model(self, cfg, dnnEstimator, design_lists):
        if cfg.run_options.mode == 'evaluate_model_report':
            return
        with open('src/Testing_models.yaml') as f:
            datamap = yaml.safe_load(f)

        evaluated_models = []
        for sol, model in enumerate(datamap):
            if 'design' in model:
                continue
            if not int(model.split('_')[-1]) in design_lists:
                continue
            print(100*"=")
            print("PYTHON : Started Synthesizing Design {} ... ".format(model))
            cfg.design_layers = datamap[model]
            given_lyrs, given_var_types = dnn_tools.pars_user_layer_defined()
            dnn_configs = self.dnn_tools.create_dnn_configs_file(given_lyrs, given_var_types)
            predicted_label = [0,0]
            cpp_segments = self.dnn_tools.create_main_cpp_file(dnn_configs, predicted_label)
            analyze_results = self.dnn_analyzer.analyze_given_model(given_lyrs)
            self.dnn_tools.create_main_header_file(cpp_segments)

            log = cfg.design_setting.Modeling['run_estimation_log']
            dnnEstimator.load_trainned_models(log=log)
            LogicSyn_estimation, topmodule_estimation, layers_estimation = self.estimate_dnn_model(cfg, dnnEstimator, given_lyrs)
            topmodule_estimation.pop('GOPS')
            topmodule_estimation.update(LogicSyn_estimation)
            start_time = self.utils.record_time()
            cfg.design_setting.design_model = 'all_models_evaluation'

            inserted_pragma = self.hls_tools.create_fixed_directive_tcl_file(
                directive_type=cfg.design_setting.syn_directive_type)
            self.hls_tools.create_syn_tcl_file(clk_period=cfg.FPGA.clock_period)
            os.chdir(cfg.paths.design_model)
            version = cfg.design_setting.vivado_version
            hls_cmd = cfg.vivado_path[sys.platform][version]['HLS']
            os.system("{} -f run_hls_syn.tcl {} > syn_report{}.log".format(hls_cmd, '', model))
            [mm, ss] = self.utils.end_and_print_time(start_time)
            print("PYTHON : Synthesis finished. Syn time : {:3d} Minutes and {:2d} Seconds".format(mm, ss))
            postSyn, power = self.hls_tools.run_vivado_implementation(0, mode=cfg.run_options.mode, print_out='', clean=True)
            temp, model_layers_name = self.hls_tools.read_single_syn_results(sol, [mm, ss], False)
            temp.update(postSyn)
            temp.update(power)
            temp['dtype'] = '{} bits'.format(cfg.design_variable_types['ker_t'])
            self.hls_tools.copy_hls_bc_files(sol_counter='', specifier=model)
            syn_results = temp[cfg.design_setting.topmodule]
            syn_results.update(postSyn)
            estimation_deviation = dnnEstimator.results_deviation(topmodule_estimation, syn_results, print_out=True)
            analyzer_deviation = self.dnn_analyzer.results_deviation([temp], print_out=True)
            layers_estimation[cfg.design_setting.topmodule] = topmodule_estimation

            analyze_exec = {}
            syn_exec = {}
            ratio1 = {}
            ratio2 = {}
            est_exec = {}
            for sublyr in list(analyze_results.keys()):
                a = lyr_map2syn[sublyr.split('_')[1]]
                if cfg.design_setting.vivado_version == 2020:
                    for ll in model_layers_name:
                        if a == 'read_input3D': syn_lbl = 'read_input3D_L1'
                        elif a == cfg.design_setting.topmodule: syn_lbl = cfg.design_setting.topmodule
                        elif a in ll:  syn_lbl = a + '_' + sublyr.split('_')[0]
                    if a not in temp.keys(): continue
                else:
                    syn_lbl = a
                analyze_exec[syn_lbl] = int(analyze_results[sublyr]['latency']) + 1  # +1 to make make > 0
                syn_exec[syn_lbl] = int(temp[syn_lbl]['latency']) + 1  # +1 to make make > 0
                est_exec[syn_lbl] = int(layers_estimation[sublyr]['latency'])
                ratio1[syn_lbl] = round(analyze_exec[syn_lbl] / syn_exec[syn_lbl], 2)
                ratio2[syn_lbl] = round(est_exec[syn_lbl] / syn_exec[syn_lbl], 2)

            for l in ['BRAM', 'LUT', 'FF', 'DSP', 'exec us', 'latency']:
                topmodule_estimation['M_'+l] = topmodule_estimation[l]
                topmodule_estimation.pop(l)

            comparison = {'math_exec': analyze_exec,
                          'syn_exec': syn_exec,
                          'est_exec': est_exec,
                          'math-syn': ratio1,
                          'est-syn': ratio2}

            design = {'ID': model,
                      'lyr_cfg': given_lyrs,
                      'syn_results': temp,
                      'analyzes': analyze_results,
                      'topmodule_estimation': topmodule_estimation,
                      'results_deviation': estimation_deviation,
                      'comparison': comparison}

            self.utils.save_a_variable('design_{}'.format(model), design)
            evaluated_models.append(design)
            os.chdir(cfg.paths.design_top)
            print(100 * "=")

        os.chdir(cfg.paths.design_top)
        self.utils.save_a_variable('evaluated_models', evaluated_models)
        return evaluated_models


# ###################################################################
# #############  create_data_for_estimation #########################

def re_structure_solutions(cfg, solutions_list, in_labels, out_labels):
    tt = list(solutions_list[0]['syn_results'].keys())
    lyr_list = tt[:tt.index(cfg.design_setting.topmodule)+1]
    new_sublayers = copy.deepcopy(lyr_list)
    #[new_sublayers.pop(i) if j.split('_')[0] in ['read', 'write'] else 0 for i, j in enumerate(lyr_list)]
    [new_sublayers.pop(i) if j.split('_')[0] in ['write', 'read', 'read_kernel3D_1'] else 0 for i, j in
     enumerate(new_sublayers)]
    [new_sublayers.pop(i) if j.split('_')[0] in ['write', 'read', 'read_kernel3D_1'] else 0 for i, j in
     enumerate(new_sublayers)]
    design_keys = tt[tt.index(cfg.design_setting.topmodule)+1:]
    new_solution_list = []
    for sol in solutions_list:
        new_sol = {}
        new_sol['lyr_cfg'] = sol['lyr_cfg']

        new_sol['solution'] = {}
        for i in ['solution', 'syn_time', 'dtype']:
            new_sol['solution'][i] = sol['syn_results'][i]

        new_sol['layers'] = {}
        for i in new_sublayers:
            new_sol['layers'][i] = sol['syn_results'][i]

        new_sol['PR'] = {}
        for i in ['LUT_PS', 'FF_PS', 'DSP_PS', 'BRAM_PS', 'P_Slice', 'P_Block', 'P_DSPs', 'P_Static', 'P_Total', 'dtype', 'Timing_PS']:
            new_sol['PR'][i] = sol['syn_results'].get(i, -1)

        for lyr_indx, lyr in enumerate(new_sublayers[:-1]):
            for lyr_cfg in in_labels:
                new_sol['layers'][lyr][lyr_cfg] = new_sol['lyr_cfg'][lyr_indx][lyr_cfg]

        new_sol['inserted_pragma'] = sol['inserted_pragma']
        new_solution_list.append(new_sol)

    return new_solution_list

def bram_utilization_updater(dnn_analyzer, cfg, solutions_list):
    top_mem_record = []
    for sol in solutions_list:
        lyr_cfg = sol['lyr_cfg']
        analyze_results = dnn_analyzer.analyze_given_model(lyr_cfg)
        temp = sol['layers'][cfg.design_setting.topmodule]
        mem_bits = {}
        Ratio = {}
        top_brams = int(temp['BRAM'])
        top_bits = analyze_results[cfg.design_setting.topmodule]['out_mem_bits']
        BRAM = {}
        for key_indx, key in enumerate(analyze_results.keys()):
            l = key
            mem_bits[l] = analyze_results[key]['out_mem_bits']
            BRAM[l] = round(top_brams * mem_bits[l]/top_bits, 2)
            Ratio[l] = round(mem_bits[l] / top_brams)  # make sure it is pointing to the last layer
        top_mem_record.append({"BRAM":BRAM, 'mem_bits':mem_bits, 'Ratio': Ratio})

    updated_solution_list = copy.deepcopy(solutions_list)
    for sol_indx, sol in enumerate(updated_solution_list):
        for lyr_indx, lyr in enumerate(list(sol['layers'].keys())[:-1]):
            pointer = list(top_mem_record[sol_indx]['BRAM'].keys())[lyr_indx]
            sol['layers'][lyr]['BRAM'] = top_mem_record[sol_indx]['BRAM'][pointer]

    return updated_solution_list

def compact_results(cfg, path, solutions_list, in_labels, out_labels, notes):
    import pandas as pd
    total_sol = len(solutions_list)
    dnn_layers = list(solutions_list[0]['layers'].keys())
    top = dnn_layers[-1]
    compacted_results = {}
    compactedResultsForTraining = {}
    for lyr in dnn_layers:
        list_lines_csv = []
        list_lines_pickle = []
        header_labels = solutions_list[0]['layers'][lyr]
        for sol in solutions_list:
            temp = {}
            if lyr in top:
                list_lines_csv.append(sol['layers'][lyr])
                list_lines_pickle.append(sol['layers'][lyr])
            else:
                for i in in_labels + out_labels:
                    temp[i] = sol['layers'][lyr][i]
                list_lines_csv.append(temp)
                list_lines_pickle.append(sol['layers'][lyr])

        csvfile = os.path.join(cfg.paths.design_top, path, '{}.csv'.format(lyr))
        df = pd.DataFrame(list_lines_csv)
        df.to_csv(csvfile, index=False, header=header_labels)
        compactedResultsForTraining[lyr] = list_lines_csv

        compacted_results[lyr] = list_lines_pickle
    compacted_results['notes'] = '\n'.join(notes)
    pickle_file = os.path.join(cfg.paths.design_top, path, 'compacked_{}solutions.pickle'.format(total_sol))
    with open(pickle_file, 'wb') as f:
        pickle.dump(compacted_results, f)

    compactedResultsForTraining['notes'] = '\n'.join(notes)
    pickle_file = os.path.join(cfg.paths.design_top, path, 'compacked_{}solutions_forTraining.pickle'.format(total_sol))
    with open(pickle_file, 'wb') as f:
        pickle.dump(compactedResultsForTraining, f)


def create_data_for_estimation(dnn_dse, dnn_analyzer, cfg, path):
    from hls_tools import collect_design_notes
    in_labels = ['w_in', 'w_out', 'lyr_in', 'lyr_out', 'w_ker', 'stride']
    out_labels = ['clock period', 'LUT', 'FF', 'DSP', 'BRAM', 'latency']
    solutions_list = dnn_dse.load_all_solutions_after_DSE(path)
    new_solution_list = re_structure_solutions(cfg, solutions_list, in_labels, out_labels)
    new_solution_list = bram_utilization_updater(dnn_analyzer, cfg, new_solution_list)
    additional_info = {'Sol_len':len(solutions_list)}
    notes = collect_design_notes(cfg, additional_info, save_path=cfg.paths.dse_report)
    compact_results(cfg, path, new_solution_list, in_labels, out_labels, notes)
    print('PYTHON : a compacted pickle file is created.')

