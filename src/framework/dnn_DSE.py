# ////////////////////////////////////////////////////////////////////////////////////////
# Copyright (c) by
# Company:  IDEA LAB, The University of Texas at Dallas
# Author :  Masoud Shahshahani
#
# Originally Create Date: Mar-5, 2020
# Project Name    : DNN_Framework
# Tool Versions   : Python +3
#
# Description     : Execute DSE_pragma, DSE_clock, DSE_dtype, DSE_universal, generate csv file
# Dependencies    : Vivado 2018 or newer, subprocess, panda
# Additional Comments:
#
# ///////////////////////////////////////////////////////////////////////////////////////

import os, shutil,sys
from shutil import copyfile
import sys, glob, json, random, pickle
import numpy as np
import random, re
from xml.dom import minidom
import subprocess
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from operator import itemgetter
from datetime import datetime
import concurrent.futures
import logging, time


cur_path = os.getcwd()
sys.path.append('/home/eng/m/mxs161831/Desktop/dnn_small')
sys.path.append(os.path.join(cur_path, 'dnn_python'))



from utils import *
from utils import beep
from hls_tools import *
from hls_tools import collect_design_notes

lyr_map2syn = {
    'dnn_LeNet':'dnn_LeNet',
    'dnn_C1P1F1':'dnn_C1P1F1',
    'dnn_C2P2F1':'dnn_C2P2F1',
    'dnn_AlexNet':'dnn_AlexNet',
    'dnn_ConvNet':'dnn_ConvNet',
    'IN':'read_input3D',
    'CONV':'conv_3DT1',
    'POOL':'ds_3DT1',
    'FC':'fc_T1'}

lyr_syn2map = {
    'dnn_LeNet':'dnn_LeNet',
    'dnn_C1P1F1':'dnn_C1P1F1',
    'dnn_C2P2F1':'dnn_C2P2F1',
    'dnn_AlexNet':'dnn_AlexNet',
    'dnn_ConvNet':'dnn_ConvNet',
    'read_input3D':'IN',
    'conv_3DT1':'CONV',
    'ds_3DT1':'POOL',
    'fc_T1':'FC'}

def data_type_gen(type, variable_types):
    data_type_list = {}
    variable_types['mid_t'] = variable_types['in_t'] + variable_types['ker_t']
    variable_types['cfg_t'] = 16
    for v_name in ['in_t', 'ker_t', 'res_t', 'mid_t', 'cfg_t']:
        if type == 'ap_int':
            data_type_list[v_name] = "typedef  ap_int<{}>   {};".format(variable_types[v_name], v_name)
        elif type == 'ap_fixed':
            data_type_list[v_name] = "typedef  ap_fixed<{},{}>   {};".format(variable_types[v_name], 4, v_name)
        else:
            data_type_list[v_name] = "typedef  int  {};".format(v_name)
    return data_type_list

def timeout_command(command, timeout):
  """call shell-command and either return its output or kill it
  if it doesn't normally exit within timeout seconds and return None"""
  import subprocess, datetime, os, time, signal
  start = datetime.datetime.now()
  process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  while process.poll() is None:
    time.sleep(0.1)
    now = datetime.datetime.now()
    if (now - start).seconds > timeout:
      os.kill(process.pid, signal.SIGKILL)
      os.waitpid(-1, os.WNOHANG)
      return None
  return process.stdout.read()


# ###################################################################
# ######################   DSE CFG    ###############################

def generate_list_of_conv(cfg):
    comb_list = {}
    org_cfg = copy.deepcopy(cfg.design_layers)
    wout_list = []
    lyrin_list = []
    lyrout_list = []
    ker_list = []
    strd_list = []
    cfg_ranges = cfg.design_setting.DSE_setting['cfg_ranges']
    wout_params = cfg_ranges['wout_range'];
    lyr_in_params = cfg_ranges['lyr_in_range'];
    lyr_out_params = cfg_ranges['lyr_out_range']
    ker_range = cfg_ranges['ker_range']
    stride_range = cfg_ranges['stride_range']
    wout_range = np.linspace(wout_params[0], wout_params[1], wout_params[2], dtype=int).tolist()
    lyr_in_range = np.linspace(lyr_in_params[0], lyr_in_params[1], lyr_in_params[2], dtype=int).tolist()
    lyr_out_range = np.linspace(lyr_out_params[0], lyr_out_params[1], lyr_out_params[2], dtype=int).tolist()
    for wout in wout_range:
        for lyrin in lyr_in_range:
            for lyrout in lyr_out_range:
                for ker in ker_range:
                    for strd in stride_range:
                        if ker > strd:
                            strd_list.append(strd)
                            ker_list.append(ker)
                            lyrout_list.append(lyrout)
                            lyrin_list.append(lyrin)
                            wout_list.append(wout)

    comb_list['w_out'] = wout_list
    comb_list['lyr_out'] = lyrout_list
    comb_list['lyr_in'] = lyrin_list
    comb_list['w_ker'] = ker_list
    comb_list['stride'] = strd_list
    print('list made')
    return comb_list


def generate_list_of_fc(cfg):
    comb_list = {}
    org_cfg = copy.deepcopy(cfg.design_layers)
    lyrin_list = []
    lyrout_list = []
    lyr_in_range = numpy.linspace(10, 500, 20, dtype=int).tolist()
    lyr_out_range = numpy.linspace(10, 500, 20, dtype=int).tolist()
    for lyrin in lyr_in_range:
        for lyrout in lyr_out_range:
            lyrout_list.append(lyrout)
            lyrin_list.append(lyrin)

    comb_list['lyr_out'] = lyrout_list
    comb_list['lyr_in'] = lyrin_list
    print('list made')
    return comb_list


def run_dse_cfg(utils, hls_tools, dnn_analyzer, dnn_tools, cfg):
    '''
    Runs DSE to explore CNN layer configurations using the user provided ranges
    :param cfg:
    :return:
    '''
    options_mode = cfg.run_options.mode
    target_layer = 1  # the DSE will change the configuration of this target layer
    lyr_cfg_analyzer_label = ['w_in', 'w_out', 'lyr_in', 'lyr_out', 'w_ker', 'stride']
    if options_mode == 'dse_cfg_report':
        return
    else:
        syn_results = []
        lyr_configs = []
        org_cfg = copy.deepcopy(cfg.design_layers)
        os.chdir(cfg.paths.design_model)
        comb_list = generate_list_of_conv(cfg)
        total_comb = len(comb_list['lyr_in'])
        random_list = random.sample(range(total_comb), cfg.design_setting.DSE_setting['solution_counts'])
        random_list.append(0)
        random_list.append(5)
        random_list.append(10)
        # random_list.append(total_comb-10)
        print('DSE CFG : Total number of combinations are {}'.format(len(random_list)))
        for sol_num, sol in enumerate(random_list):
            cfg.design_layers = copy.deepcopy(org_cfg)

            for label in comb_list.keys():
                if label == 'lyr_in':
                    cfg.design_layers[target_layer - 1]['lyr_out'] = comb_list[label][sol]
                elif label == 'w_out':
                    cfg.design_layers[target_layer - 1][label] = comb_list[label][sol]
                else:
                    cfg.design_layers[target_layer][label] = comb_list[label][sol]

            given_lyrs, given_var_types = dnn_tools.pars_user_layer_defined()
            dnn_configs = dnn_tools.create_dnn_configs_file(given_lyrs, given_var_types)
            analyze_results = dnn_analyzer.analyze_given_model(given_lyrs)
            fname = os.path.join(cfg.paths.dse_analyzes, 'analyzes{}'.format(sol))
            dnn_analyzer.save_overall_report(fname=fname, plot=False)

            print_str = ''
            for cfg_label in lyr_cfg_analyzer_label:
                print_str = print_str + ', {}={}'.format(cfg_label, dnn_configs[target_layer][cfg_label])

            print(100 * "=")
            print("DSE #{}/{} sol={} ,lyr={} {}   is started".format(sol_num, len(random_list), sol, target_layer,
                                                                     print_str))

            inserted_pragma = hls_tools.create_fixed_directive_tcl_file(
                directive_type=cfg.design_setting.syn_directive_type)
            hls_tools.create_syn_tcl_file(clk_period=cfg.FPGA.clock_period)
            start_time = utils.record_time()
            os.chdir(cfg.paths.design_model)

            version = cfg.design_setting.vivado_version
            hls_cmd = cfg.vivado_path[sys.platform][version]['HLS']
            os.system("{} -f run_hls_syn.tcl > syn_report{}.log".format(hls_cmd, ''))

            [mm, ss] = utils.end_and_print_time(start_time)
            beep('dse')
            print("PYTHON : DSE on cfg: Synthesis finished. Syn time : {:3d} Minutes and {:2d} Seconds".format(mm, ss))
            temp, model_layers_name = hls_tools.read_single_syn_results(sol, [mm, ss], False)
            if temp['syn_status'] == 'failed':
                continue
            temp['solution'] = sol
            postSyn, power = hls_tools.run_vivado_implementation(0, mode=options_mode, print_out='', clean=True)
            hls_tools.copy_hls_bc_files(sol)
            temp.update(postSyn)
            temp.update(power)
            temp['dtype'] = '{} bits'.format(cfg.design_variable_types['ker_t'])
            temp['dse_lyr'] = target_layer

            analyze_exec = {}
            syn_exec = {}
            ratio = {}
            for sublyr in list(analyze_results.keys()):
                a = lyr_map2syn[sublyr.split('_')[1]]
                for ll in model_layers_name:
                    if a in ll:
                        syn_lbl = ll
                    else:
                        continue
                    analyze_exec[syn_lbl] = int(analyze_results[sublyr]['latency']) + 1  # +1 to make make > 0
                    syn_exec[syn_lbl] = int(temp[syn_lbl]['latency']) + 1  # +1 to make make > 0
                    ratio[syn_lbl] = round(analyze_exec[syn_lbl] / syn_exec[syn_lbl], 2)
            comparison = {'math_exec': analyze_exec,
                          'syn_exec': syn_exec,
                          'math-syn': ratio}

            design = {'ID': sol,
                      'lyr_cfg': given_lyrs,
                      'syn_results': temp,
                      'analyzes': analyze_results,
                      'inserted_pragma': inserted_pragma,
                      'comparison': comparison}

            copy_solution_files(cfg, sol)
            utils.save_a_variable('design{}'.format(sol), design)
            syn_results.append(temp)
            lyr_configs.append(given_lyrs)
            utils.save_a_variable('lyr_configs', lyr_configs)
            utils.save_a_variable('syn_results', syn_results)
        os.chdir(cfg.paths.design_top)
        utils.save_a_variable('model_layers_name', model_layers_name)
    return syn_results, lyr_configs, model_layers_name

def run_dse_clk_pragma_cfg(cfg):
    '''
    Runs DSE over many different pragma combinations
    For each pragma set, explore the CNN layer configurations using the user provided ranges
    :param cfg:
    :return:
    '''
    target_layer = 1  # the DSE will change the configuration of this target layer
    lyr_cfg_analyzer_label = ['w_in', 'w_out', 'lyr_in', 'lyr_out', 'w_ker', 'stride']
    if options.mode == 'dse_clk_pragma_cfg_report':
        return
    else:

        clk_range = cfg.design_setting.DSE_setting['clock_range']
        freq_list = np.linspace(clk_range['min'], clk_range['max'], clk_range['samples']).astype(int)
        period_list = np.around(1000 / freq_list, 1)

        # setup CNN layer configs to explore
        syn_results = []
        lyr_configs = []
        org_cfg = copy.deepcopy(cfg.design_layers)
        os.chdir(cfg.paths.design_model)

        for network in cfg.design_setting.DSE_setting['cfg_ranges']['networks'].keys():
            cfg.design_setting.design_model = network
            parsed_cfg = gen_configs.parse_yaml_design_arguments()
            cfg.design_layers = parsed_cfg['design_layers']
            cfg.pragmas = parsed_cfg['pragmas']

            # setup pragmas to explore
            [fixed_pragmas, variable_pragmas, total_comb] = hls_tools.pars_dnn_design_pragmas(cfg.pragmas['variable'])
            selection_type = cfg.design_setting.DSE_setting['directive_selection']
            selected_pragmas = dnn_dse.create_dse_directive_tcl_file(selection_type, fixed_pragmas, variable_pragmas,
                                                                     total_comb)

            comb_list = generate_list_of_conv(cfg)
            total_comb = len(comb_list['lyr_in'])
            config_list = random.sample(range(total_comb), cfg.design_setting.DSE_setting['config_count'])

            total_count = len(config_list) * len(selected_pragmas) * len(period_list)
            print(40*'='+" Design : {} ".format(network)+40*'=')
            print('DSE_PRAGMA_CFG_Clock: Begin DSE. Total number of combinations are {}'.format(total_count))
        # For each set in selected_pragmas, run synthesis for all configs in config_list
            for clk_indx, clk_period in enumerate(period_list):
                for pragma_ind, pragma_sol in enumerate(selected_pragmas):
                    # setup this pragma_sol for synthesis
                    copyfile(os.path.join(cfg.paths.directive_list, "solution_{}.tcl".format(pragma_sol)), cfg.files.DirectiveFile)
                    hls_tools.create_syn_tcl_file(clk_period=clk_period)
                    for cfg_ind, cfg_sol in enumerate(config_list):
                        design_name = '{}_{}_{}'.format(pragma_sol, clk_period, cfg_sol)
                        # setup this cfg_sol for synthesis
                        cfg.design_layers = copy.deepcopy(org_cfg)
                        for label in comb_list.keys():
                            if label == 'lyr_in':
                                cfg.design_layers[target_layer - 1]['lyr_out'] = comb_list[label][cfg_sol]
                            elif label == 'w_out':
                                cfg.design_layers[target_layer - 1][label] = comb_list[label][cfg_sol]
                            else:
                                cfg.design_layers[target_layer][label] = comb_list[label][cfg_sol]

                        given_lyrs, given_var_types = dnn_tools.pars_user_layer_defined()
                        dnn_configs = dnn_tools.create_dnn_configs_file(given_lyrs, given_var_types)
                        analyze_results = dnn_analyzer.analyze_given_model(given_lyrs)
                        fname = os.path.join(cfg.paths.dse_analyzes, 'analyzes{}'.format(design_name))
                        dnn_analyzer.save_overall_report(fname=fname, plot=False)

                        # print info bar
                        print_str = ''
                        for cfg_label in lyr_cfg_analyzer_label:
                            print_str = print_str + ', {}={}'.format(cfg_label, dnn_configs[target_layer][cfg_label])
                        print(100 * "=")
                        current_indx = clk_indx*len(selected_pragmas)*len(config_list)+ pragma_ind*len(config_list) + cfg_ind + 1
                        print("DSE_PRAGMA_CFG_CLK: {}/{} design{} is started.\nlyr={} {}".format(
                            current_indx , total_count, design_name, target_layer, print_str))

                        # run the HLS synthesis
                        start_time = utils.record_time()
                        syn_status = hls_tools.run_hls_synth('syn', cfg.design_setting.syn_timeout, 'silent',
                                                                  clean=True, sol=pragma_sol)
                        [mm, ss] = utils.end_and_print_time(start_time)
                        if not syn_status:
                            continue

                        temp, model_layers_name = hls_tools.read_parallel_syn_results(pragma_sol, [mm, ss], False)
                        temp['solution'] = design_name
                        postSyn, power = hls_tools.run_vivado_implementation(pragma_sol, mode='dse_pragma', print_out='silent', clean=True)
                        temp.update(postSyn)
                        temp.update(power)
                        temp['dtype'] = '{} bits'.format(cfg.design_variable_types['ker_t'])
                        temp['dse_lyr'] = target_layer

                        analyze_exec = {}
                        syn_exec = {}
                        ratio = {}
                        for sublyr in list(analyze_results.keys()):
                            a = lyr_map2syn[sublyr.split('_')[1]]
                            for ll in model_layers_name:
                                if a in ll:
                                    syn_lbl = ll
                                else:
                                    continue
                                analyze_exec[syn_lbl] = int(analyze_results[sublyr]['latency']) + 1  # +1 to make make > 0
                                syn_exec[syn_lbl] = int(temp[syn_lbl]['latency']) + 1  # +1 to make make > 0
                                ratio[syn_lbl] = round(analyze_exec[syn_lbl] / syn_exec[syn_lbl], 2)
                        comparison = {'math_exec': analyze_exec,
                                      'syn_exec': syn_exec,
                                      'math-syn': ratio}

                        design = {'ID': design_name,
                                  'lyr_cfg': given_lyrs,
                                  'syn_results': temp,
                                  'analyzes': analyze_results,
                                  'inserted_pragma': cfg.dse_pragmas[pragma_sol],
                                  'comparison': comparison}
                        specifier = '{}_{}'.format(clk_period, cfg_sol)
                        copy_solution_files(cfg, pragma_sol, specifier=specifier)
                        hls_tools.copy_hls_bc_files(pragma_sol, specifier=specifier)
                        utils.save_a_variable('design{}'.format(design_name), design)
                        if cfg.design_setting.DSE_setting['remove_hls_run_directories'] and \
                                    os.path.exists(os.path.join(cfg.paths.design_model, 'hls{}'.format(pragma_sol))):
                                    shutil.rmtree(os.path.join(cfg.paths.design_model, 'hls{}'.format(pragma_sol)))

                        syn_results.append(temp)
                        lyr_configs.append(given_lyrs)
                        utils.save_a_variable('lyr_configs', lyr_configs)
                        utils.save_a_variable('syn_results', syn_results)
        dnn_dse.copy_all_design_sourcefile()
        os.chdir(cfg.paths.design_top)
    return syn_results, lyr_configs, model_layers_name


class dnn_dse:
    def __init__(self,cfg):
        self.cfg = cfg
        self.hls_tools = hls_tools(cfg)
        self.utils = utils(cfg)
        self.ann = []

    def replace_a_variable_value_in_file(self,fname ,VAR, VAL):
        with open(fname, 'r+') as f:
            text = f.readlines()
            new_text = []
            for linenum, line in enumerate(text):
                splitted = line.split()
                if VAR in splitted:
                    indx = splitted.index(VAR)
                    splitted[indx+1] = str(VAL)
                    line = ' '.join(splitted)
                new_text.append(line.rstrip())
        f.close()
        self.utils.save_list_to_file(fname, new_text)

    def update_dtype_in_cfg_file(self, filename, new_var_types):
        with open(filename, 'r+') as f:
            text = f.readlines()
            new_text = []
            for linenum, line in enumerate(text):
                if 'typedef' in line:
                    a = line.split()
                    tmp = a[-1].strip(';')
                    if tmp in new_var_types:
                        line = new_var_types[tmp]
                new_text.append(line.rstrip())
        f.close()
        self.utils.save_list_to_file(filename, new_text)

    def copy_all_json_files(self):
        dest_path = os.path.join(self.cfg.paths.dse_report, 'directive_solution_lists')
        if not os.path.exists(dest_path):
            shutil.copytree(self.cfg.paths.directive_list, dest_path)

    def copy_all_design_sourcefile(self):
        extensions = ['h', 'cpp']
        for extension in extensions:
            src_files = glob.iglob(os.path.join(self.cfg.paths.design_model, '*.{}'.format(extension)))
            for file in src_files:
                if os.path.isfile(file):
                    shutil.copy(file, self.cfg.paths.dse_report)

    def plot_dse(self, design_solutions, plot_subs, plot_top):
        if plot_top == True:
            colors = cm.rainbow(np.linspace(0, 1, len(plot_subs)))

            # Check if all submodules are present in pickle file
            available_modules = plot_subs
            for i in plot_subs:
                for k in design_solutions:
                    if k['syn_status'] == 'passed':
                        if i not in k.keys():
                            print("Error:", i, "is not there in pickle file of design", design_solutions.index(k))
                            available_modules.remove(i)
                    else:
                        del design_solutions[design_solutions.index(k)]
            arr_latency = np.zeros((0, len(design_solutions)))
            arr_total = np.zeros((0, len(design_solutions)))
            ##Get the list of module and their latecy and total %
            for m in available_modules:
                res = list(map(itemgetter(m), design_solutions))  # all features of each m i.e. dse_module

                res_1 = list(map(itemgetter('exec us'), res))
                arr_1 = np.around(np.array(res_1, dtype=np.float) , decimals=1)
                arr_latency = np.vstack([arr_latency, arr_1])

                res_2 = list(map(itemgetter('Total %'), res))
                arr_2 = np.around(np.array(res_2, dtype=np.float), decimals=1)
                arr_total = np.vstack([arr_total, arr_2])

            arr_latency = arr_latency.T
            arr_total = arr_total.T
            text = []
            ann = []
            xy_list=[]
            ##plotting the graph and annotation

            for count, mod in enumerate(available_modules):
                fig, ax = plt.subplots()
                ax.scatter(arr_latency[:, count], arr_total[:, count], label=mod,
                                color=colors[count])
                for i in range(len(arr_latency)):
                    for j in range(len(arr_latency)):
                        if (arr_latency[i][count] == arr_latency[j][count] and arr_total[i][count] ==
                                arr_total[j][count]):
                            text.append(" {} ".format(j))
                    xy = (arr_latency[i][count], arr_total[i][count])
                    xy_list.append(xy)

                    ann.append(
                        #ax.annotate(','.join(text), xy=(arr_latency[i][count], arr_total[i][count])))
                        ax.annotate(text[0], xy=(arr_latency[i][count], arr_total[i][count])))
                    text.clear()
                ax.set_title('Total design solutions : {}'.format(len(design_solutions)))
                ax.set_ylabel('Total Hardware utilization (%)')
                ax.set_xlabel('Total execution time (us)')
                ax.grid(b=True, which='major')
                #self.utils.save_fig(mod, fig_extension="svg")
                self.utils.save_fig(mod, fig_extension="pdf")

            #self.bnext = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Hide')
            #self.bnext.on_clicked(self.show)
            #plt.show(ann)
        return

    def plot_evaluated_models(self, design_names, solutions_list):
        plot_type = {'math_exec': '-b', 'syn_exec': '-g', 'est_exec': '-r', 'math-syn': 'b', 'est-syn': '-r'}
        rgb = ['#1f77b4', 'b','g','r','c','m','y','k']
        comp_list = list(solutions_list[0]['comparison'].keys())
        #layers = list(solutions_list[0]['comparison'][comp_list[0]])
        comp_data = {}
        for lyr in ['read', 'conv', 'ds', 'fc', 'dnn']:
            comp_data[lyr] = {}
            for comp in comp_list:
                comp_data[lyr][comp] = []
                for sol_num, sol in enumerate(solutions_list):
                    sublyrs = list(sol['comparison']['math_exec'].keys())
                    for l in list(sublyrs):
                        if lyr in l:
                            comp_data[lyr][comp].append(sol['comparison'][comp][l])

        ## ---------------  plot latency analyzes --------------------------------
        for lyr in comp_data.keys():
            fig, ax = plt.subplots(1)
            fig.set_size_inches(8, 8)
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=0.005, hspace=0.005)
            fig_name = '{}/figures/{}_latency.jpg'.format(self.cfg.paths.dse_report, lyr)
            for comp_num, comp in enumerate(['math_exec', 'syn_exec', 'est_exec']):
                ax.plot(comp_data[lyr][comp], plot_type[comp], label=comp)
            ax.legend(loc='upper right', fontsize='small')
            ax.set_xlabel('Design number')
            ax.set_ylabel('Latency')
            ax.set_title(lyr)
            plt.savefig(fig_name, dpi=200)
            plt.close(fig)

        for lyr in comp_data.keys():
            fig, ax = plt.subplots(1)
            fig.set_size_inches(8, 8)
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=0.005, hspace=0.005)
            fig_name = '{}/figures/{}_ratio.jpg'.format(self.cfg.paths.dse_report, lyr)
            maxx = 0
            for comp_num, comp in enumerate(['math-syn', 'est-syn']):
                ax.plot(comp_data[lyr][comp], plot_type[comp], label=comp)
                #maxx = max(max(comp_data[lyr][comp]), maxx)
            ax.legend(loc='upper right', fontsize='small')
            ax.set_xlabel('Design number')
            ax.set_ylabel('Latency')
            ax.set_title(lyr)
            #ax.set_ylim([0, maxx*1.2])
            #plt.yticks(np.arange(0, maxx*1.2, 1.0))
            plt.savefig(fig_name, dpi=200)
            plt.close(fig)

        ## ---------------  plot error --------------------------------
        er_data = {}
        for er in solutions_list[0]['results_deviation'].keys():
            if er[-1] == '%':
                er_data[er] = []
                for sol in solutions_list:
                    er_data[er].append(sol['results_deviation'][er])
        er_data.pop('er_latency %')

        fig, ax = plt.subplots(1)
        fig.set_size_inches(8, 8)
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=0.005, hspace=0.005)
        fig_name = '{}/figures/Estimation_error%.jpg'.format(self.cfg.paths.dse_report)
        maxx=0
        for er_n, er in enumerate(er_data.keys()):
            #maxx = max(max(er_data[er]), maxx)
            ax.plot(list(design_names.values()), er_data[er], label=er.split('_')[1][:-2], c=rgb[er_n], marker='.', ls='-')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(er_data), fancybox=False)
        ax.set_xlabel('Design number')
        ax.set_ylabel('Percentage of Estimation Error')
        #ax.set_ylim([0, maxx * 1.2])
        #plt.yticks(np.arange(0, maxx * 1.1, 1.0))
        plt.savefig(fig_name, dpi=200)
        plt.close(fig)

    def show(self, ann, event):
        if ann[0].get_visible():
            [ann[i].set_visible(False) for i in range(self.len)]
            self.bnext.label.set_text("Show")
            plt.draw()
        else:
            [ann[i].set_visible(True) for i in range(self.len)]
            self.bnext.label.set_text("Hide")
            plt.draw()

    def create_dse_excel_report(self, design_solutions, layers, lyr_configs=[], sort1_by='latency', sort2_by='DSP'):
        for lyrN, layer in enumerate(layers):
            design_solutions_passed = []
            for solution in design_solutions:
                if solution['syn_status'] == 'passed':
                    if not layer in solution.keys():
                        print('PYTHON : DSE: {} module is not in one of the solutions'.format(layer))
                        return
                    else:
                        design_solutions_passed.append(solution)

        sorted_solutions = sorted(design_solutions_passed, key=lambda x: (float(x[self.cfg.design_setting.topmodule][sort1_by]), int(x[self.cfg.design_setting.topmodule][sort2_by])))

        lyr_labels = list(sorted_solutions[0][self.cfg.design_setting.topmodule].keys())
        aa = list(sorted_solutions[0].keys())
        design_labels = aa[aa.index(self.cfg.design_setting.topmodule)+1:]
        sublayers = aa[:aa.index(self.cfg.design_setting.topmodule)+1]
        new_sublayers = sublayers # to remove read write functions from the report
        [new_sublayers.pop(i) if j.split('_')[0] in ['read', 'write'] else 0 for i, j in enumerate(sublayers)]
        for lyrN, sublyr in enumerate(new_sublayers):
            filename = os.path.join(self.cfg.paths.dse_report, "dse_{}.csv".format(sublyr))
            f = open(filename, "w+")
            for item in design_labels + lyr_labels:
                f.write(item + ',')
            f.write('\n')

            for indx, solution in enumerate(sorted_solutions, start=0):
                if solution['syn_status'] == 'failed':
                    f.write(solution['syn_status'] + ',')
                    f.write(solution['syn_time'] + ',')
                else:
                    for i in design_labels:
                        f.write(str(solution[i]) + ',')
                    for key in lyr_labels:
                        if key in solution[sublyr]:
                            f.write(str(solution[sublyr][key]) + ',')
                        else:
                            f.write('NA ,')

                f.write('\n')
            f.close()
            print('\nPYTHON : DSE: Summary of {} {} DSE is created in the report directory!'.format(sublyr,indx))

    def load_all_solutions_after_DSE(self, source_path):
        path = os.path.join(self.cfg.paths.design_top,source_path, 'pickles')
        solutions_list = []
        for design_file in glob.glob('{}/design*.pickle'.format(path)):
            dbfile = open(design_file, 'rb')
            design = pickle.load(dbfile)
            solutions_list.append(design)
        return solutions_list

    def create_dse_excel_report_new(self, target_layers='top', add_cfg_labels=False):
        solutions_list = []
        rpt_ref_sol = 0
        cfg_labels = ['w_in', 'w_out', 'lyr_in', 'lyr_out', 'w_ker', 'stride']
        pickle_files_path = os.path.join(self.cfg.paths.dse_pickles, 'design*.pickle')
        for design_file in glob.glob(pickle_files_path):
            dbfile = open(design_file, 'rb')
            design = pickle.load(dbfile)
            solutions_list.append(design)
        #lyr_labels = list(solutions_list[rpt_ref_sol]['syn_results'][self.cfg.design_setting.topmodule].keys())
        aa = list(solutions_list[rpt_ref_sol]['syn_results'].keys())
        design_labels = aa[aa.index(self.cfg.design_setting.topmodule)+1:]
        sublayers = aa[:aa.index(self.cfg.design_setting.topmodule)+1]
        for indx, j in enumerate(sublayers):
            if j.split('_')[0] in ['read', 'write']:
                sublayers.pop(indx)
        new_sublayers = sublayers # to remove read write functions from the report
        additional_label1 = []
        additional_label2 = []
        comp_labels = []
        for key in solutions_list[rpt_ref_sol].keys():
            if key in ['topmodule_estimation']:
                additional_label1 = list(solutions_list[rpt_ref_sol][key].keys())
            elif key in ['results_deviation']:
                additional_label2 = list(solutions_list[rpt_ref_sol][key].keys())
            elif key in ['comparison']:
                comp_labels = list(solutions_list[rpt_ref_sol][key].keys())

        # labels for Power and Area Efficiency metrics
        efficiency_labels = ['P_eff', 'A_eff']

        if target_layers == 'top':
            new_sublayers = [self.cfg.design_setting.topmodule]
        else:
            [new_sublayers.pop(i) if j.split('_')[0] in ['read', 'write'] else 0 for i, j in enumerate(sublayers)]

        # Write to the CSV report files for the CNN, and for each layer in the CNN.
        for lyrN, sublyr in enumerate(new_sublayers):
            filename = os.path.join(self.cfg.paths.dse_report, "dse_{}.csv".format(sublyr))
            f = open(filename, "w+")
            if sublyr != self.cfg.design_setting.topmodule and add_cfg_labels:
                for i in cfg_labels:
                    f.write(i + ',')
            for item in design_labels + list(solutions_list[rpt_ref_sol]['syn_results'][sublyr].keys()) + additional_label1 + additional_label2 + comp_labels + efficiency_labels:
                f.write(item + ',')
            f.write('\n')

            for indx, solution in enumerate(solutions_list, start=0):
                if solution['syn_results']['syn_status'] == 'failed':
                    f.write(solution['syn_results']['syn_status'] + ',')
                    f.write(solution['syn_results']['syn_time'] + ',')
                else:

                    if sublyr != self.cfg.design_setting.topmodule and add_cfg_labels:
                        for i in cfg_labels:
                            f.write(str(solution['lyr_cfg'][lyrN+1].get(i, 'NR')) + ',')

                    for i in design_labels:
                        f.write(str(solution['syn_results'].get(i, 'NR')) + ',')

                    for key in solution['syn_results'][sublyr].keys():
                        f.write(str(solution['syn_results'][sublyr].get(key, 'NR')) + ',')

                    for key in additional_label1:
                        f.write(str(solution['topmodule_estimation'].get(key, 'NR')) + ',')

                    for key in additional_label2:
                        f.write(str(solution['results_deviation'].get(key, 'NR')) + ',')

                    for key in comp_labels:
                        f.write(str(solution['comparison'][key][sublyr]) + ',')

                    # Compute efficiency metrics
                    GOPS = solution['syn_results'][sublyr].get('GOPS', 0)
                    P_Total = solution['syn_results'].get('P_Total', 1)
                    used_DSP = solution['syn_results'].get('DSP_PS', 1)
                    used_BRAM = solution['syn_results'].get('BRAM_PS', 1)

                    if 'NR' not in [GOPS, P_Total]:
                        P_eff = 1000*float(GOPS) / float(P_Total)
                        f.write(str(P_eff) + ',')

                    if 'NR' not in [GOPS, used_DSP, used_BRAM]:
                        A_eff = float(GOPS) * float(self.cfg.FPGA.DSP)/float(used_DSP) * float(self.cfg.FPGA.BRAM) / float(used_BRAM)
                        f.write(str(A_eff) + ',')

                f.write('\n')
            f.close()
            print('PYTHON : DSE: Summary of {} {} DSE is created in the report directory!'.format(sublyr,indx+1))

        additional_info = {'Sol_len': len(solutions_list)}
        notes = collect_design_notes(self.cfg, additional_info, save_path=self.cfg.paths.dse_report)
        return solutions_list

    def create_random_directive_tcl_file(self, design_counter, fixed_pragmas, variable_pragmas):
        tcl_lines = []
        tcl_lines.append("############################################################")
        tcl_lines.append("## This file is generated automatically by dnn tool.")
        tcl_lines.append("############################################################")
        tcl_lines.append("# --------------------  fixed pragmas are below ------------------ ")
        for item in fixed_pragmas:
            tcl_lines.append(item)

        tcl_lines.append("# ---------------------------------------------------------------- ")
        tcl_lines.append("# ------------------ variable pragmas are below ------------------ ")
        for pragma in variable_pragmas:
            directive_indx = random.randint(0, len(pragma)-1)
            tcl_lines.append(pragma[directive_indx])


        filename =  os.path.join(self.cfg.paths.directive_list, "solution_{}.tcl".format(design_counter))
        self.utils.save_list_to_file(filename, tcl_lines)

    def create_dse_directive_tcl_file(self,pragma_gen_style,fixed_pragmas,variable_pragmas, total_comb):
        ctrl_intf = {'hs': "set_directive_interface -mode ap_ctrl_hs \"{}\"".format(self.cfg.design_setting.topmodule),
        'none': "set_directive_interface -mode ap_ctrl_none \"{}\"".format(self.cfg.design_setting.topmodule),
        'axi':"set_directive_interface -mode s_axilite \"{}\"".format(self.cfg.design_setting.topmodule)}

        fixed_lines = []
        fixed_lines.append("############################################################")
        fixed_lines.append("## This file is generated automatically by dnn tool.")
        fixed_lines.append("############################################################")
        fixed_lines.append("# --------------------  fixed pragmas are below ------------------ ")
        fixed_lines.append(ctrl_intf.get(self.cfg.design.module_controller,''))
        for intf in self.cfg.interface[self.cfg.design.data_interface]:
            fixed_lines.append(intf)

        for item in fixed_pragmas:
            fixed_lines.append(item)
        fixed_lines.append("# ---------------------------------------------------------------- ")
        fixed_lines.append("# ------------------ variable pragmas are below ------------------ ")

        numlst = list(range(total_comb))
        if pragma_gen_style == 'random':
            random.shuffle(numlst)
        total_dse_solutions = min(300, total_comb, self.cfg.design_setting.DSE_setting['solution_counts'])
        selected_solutions = numlst[:total_dse_solutions]
        pragma_indx_list = []
        repeat = 1
        temp = total_comb
        for indx, pragma in enumerate(variable_pragmas):
            temp = round(temp / len(pragma))
            a = np.array([[np.ones(temp, int) * i for i in range(len(pragma))] for i in range(repeat)])
            repeat = repeat * len(pragma)
            pragma_indx_list.append(a.ravel().tolist())

        pragma_indx_list = np.array(pragma_indx_list).T
        dse_pragmas = {}
        for design in selected_solutions:
            tcl_lines = fixed_lines.copy()
            for indx, pragma in enumerate(pragma_indx_list[design]):
                tcl_lines.append(variable_pragmas[indx][pragma])

            if self.cfg.design.dataflow:
                tcl_lines.append('\nset_directive_dataflow    \"{}\"'.format(self.cfg.design_setting.topmodule))
            dse_pragmas[design] = tcl_lines
            filename = os.path.join(self.cfg.paths.directive_list, "solution_{}.tcl".format(design))
            self.utils.save_list_to_file(filename, tcl_lines)
        self.cfg.dse_pragmas = dse_pragmas
        print("PYTHON : DSE: Pragma style = {}\n\tTotal possible directive combinations  = {}".format(pragma_gen_style,total_comb))
        print("\tTotal requested combinations  = {}.".format( total_dse_solutions))
        print("\t{} directive tcl files are created!".format(total_dse_solutions))
        return selected_solutions

    def read_dse_syn_results(self, syn_exec_time):
        solution_syn_list = []
        for solNum, file in enumerate(glob.glob(self.cfg.paths.directive_list + "/" + "*.json")):
            try:
                with open(file) as json_file:
                    json_data = json.load(json_file)
                    passed_sol = self.extract_hls_json_info(json_data)
                    passed_sol['solution'] = re.findall(r'\w+',file)[-2]
                    passed_sol['syn_status'] = 'passed'
                    passed_sol['syn_time'] = '{:3d}:{:2d}'.format(syn_exec_time[solNum][0],syn_exec_time[solNum][1])
                    solution_syn_list.append(passed_sol)
                    json_file.close()
            except IOError:
                print("PYTHON : can't open {} file".format(file))
        return solution_syn_list

# ===================================================================================
# ========================    DSE functions    ======================================
# ===================================================================================
    def remove_all_synthesize_results(self, total_dse_solutions):
        for i in range(total_dse_solutions):
            syn_path = os.path.join(self.cfg.paths.design_model, 'hls{}'.format(i))
            shutil.rmtree(syn_path)

    def parallel_dse_pragma(self, sol):
        syn_path = os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol), self.cfg.design_setting.solution_name)
        start_time = self.utils.record_time()
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("PYTHON : DSE on pragmas: Synthesis of design {} is started at {}.".format(sol, dt_string))
        copyfile(self.cfg.paths.directive_list + "\solution_{}.tcl".format(sol), self.cfg.files.DirectiveFile)
        self.hls_tools.create_syn_tcl_file(clk_period=self.cfg.FPGA.clock_period)
        os.chdir(self.cfg.paths.design_model)
        syn_status = self.hls_tools.run_hls_synth('syn', self.cfg.design_setting.syn_timeout, 'silent', clean=True, sol=sol)
        if not syn_status:
            [mm, ss] = self.utils.end_and_print_time(start_time)
            print("PYTHON : DSE_pragma: Synthesis of design {} is failed. Syn time :{:3d}:{:2d}".format(sol, mm, ss))
            beep('syn')
            time.sleep(5)
            temp, model_layers_name = self.hls_tools.read_parallel_syn_results(sol, [mm, ss], False)
            temp['dtype'] = '{} bits'.format(self.cfg.design_variable_types['ker_t'])
        else:
            beep('syn')
            postSyn, power = self.hls_tools.run_vivado_implementation(sol, mode='dse_pragma', print_out='silent', clean=True)
            [mm, ss] = self.utils.end_and_print_time(start_time)
            print("PYTHON : DSE_pragma: Synthesis of design {} is finished. Syn time :{:3d}:{:2d}".format(sol, mm, ss))
            temp, model_layers_name = self.hls_tools.read_parallel_syn_results(sol, [mm, ss], False)
            temp['dtype'] = '{} bits'.format(self.cfg.design_variable_types['ker_t'])
            temp.update(postSyn)
            temp.update(power)

        os.chdir(self.cfg.paths.design_top)
        copy_solution_files(self.cfg, sol)
        self.hls_tools.copy_hls_bc_files(sol)
        if self.cfg.design_setting.DSE_setting['remove_hls_run_directories'] and \
                os.path.exists(os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol))):
            shutil.rmtree(os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol)))
        self.utils.save_a_variable('solution_{}'.format(sol), temp)
        return temp

    def sequential_dse_pragma(self, selected_solutions, clk_period):
        dse_pragma = []
        num_solutions = len(selected_solutions)
        for sol_num, sol in enumerate(selected_solutions):
            sol_num = sol_num+1 #add 1 to account for starting at index 0
            syn_path = os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol), self.cfg.design_setting.solution_name)
            start_time = self.utils.record_time()
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print("\nPYTHON : DSE {}/{}: Synthesis of design {} is started at {}.".format(sol_num, num_solutions, sol, dt_string))
            copyfile(os.path.join(self.cfg.paths.directive_list, "solution_{}.tcl".format(sol)), self.cfg.files.DirectiveFile)
            self.hls_tools.create_syn_tcl_file(clk_period=self.cfg.FPGA.clock_period)
            syn_status = self.hls_tools.run_hls_synth('syn', self.cfg.design_setting.syn_timeout, 'silent', clean=True, sol=sol)
            if not syn_status:
                [mm, ss] = self.utils.end_and_print_time(start_time)
                print("PYTHON : FAILED Synthesis DSE {}/{}: Synthesis of design {} is failed. Syn time :{:3d}:{:2d}".format(
                    sol_num, num_solutions, sol, mm, ss))
                beep('syn')
                time.sleep(5)
                temp, model_layers_name = self.hls_tools.read_parallel_syn_results(sol, [mm, ss], False)
                temp['dtype'] = '{} bits'.format(self.cfg.design_variable_types['ker_t'])
            else:
                beep('syn')
                postSyn, power = self.hls_tools.run_vivado_implementation(sol, mode='dse_pragma', print_out='silent', clean=True)
                [mm, ss] = self.utils.end_and_print_time(start_time)
                print("PYTHON : COMPLETED Synthesis DSE {}/{}: Synthesis of design {} is finished. Syn time :{:3d}:{:2d}".format(sol_num, num_solutions, sol, mm, ss))
                temp, model_layers_name = self.hls_tools.read_parallel_syn_results(sol, [mm, ss], False)
                temp['dtype'] = '{} bits'.format(self.cfg.design_variable_types['ker_t'])
                temp.update(postSyn)
                temp.update(power)

            design = {'ID': sol,
                      'lyr_cfg': self.cfg.design_layers,
                      'syn_results': temp,
                      'analyzes': self.cfg.analyze_results,
                      'inserted_pragma': self.cfg.dse_pragmas[sol]}
            copy_solution_files(self.cfg, sol)
            self.hls_tools.copy_hls_bc_files(sol)
            if self.cfg.design_setting.DSE_setting['remove_hls_run_directories'] and \
                    os.path.exists(os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol))):
                shutil.rmtree(os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol)))
            dse_pragma.append(design)
            os.chdir(self.cfg.paths.design_top)
            self.utils.save_a_variable('design_{}'.format(sol), design)
            self.utils.save_a_variable('dse_pragma', dse_pragma)
        return dse_pragma

    def load_parallel_dse_pragma_solutions(self):
        solution_syn_list = []
        sol_pickle_file_list = self.utils.list_files_with_ext(self.cfg.paths.dse_report, 'pickle')
        for i in sol_pickle_file_list:
            with open(os.path.join(self.cfg.paths.dse_report, i), 'rb') as f:
                solution_syn_list.append(pickle.load(f))
        return solution_syn_list

    def run_dse_pragma(self, design_name, options):
        '''
        Perform Design Space Exploration by performing a C synthesis on a designated number of random pragma combinations
        :param design_name:
        :param options:
        :return:
        '''
        if options.mode == 'dse_pragma_report':
            print("PYTHON : DSE_pragma skipped...generating reports only")
            return '', ''
        else:
            if design_name == self.cfg.design_setting.topmodule:
                [fixed_pragmas, variable_pragmas, total_comb] = self.hls_tools.pars_dnn_design_pragmas(
                self.cfg.pragmas['variable'])
            else:
                [fixed_pragmas, variable_pragmas, total_comb] = self.hls_tools.pars_DA_design_pragmas(
                    self.cfg.pragmas['variable'])
            selection_type = self.cfg.design_setting.DSE_setting['directive_selection']
            selected_solutions = self.create_dse_directive_tcl_file(selection_type, fixed_pragmas, variable_pragmas, total_comb)
            format = "%(asctime)s: %(message)s"
            logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
            max_workers = min(len(selected_solutions),self.cfg.design_setting.DSE_setting['max_parallel_syn'])
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            if max_workers == 1:
                print('PYTHON : {} HLS Synthesis will be run in sequence. Starting at {}'.format(len(selected_solutions), dt_string))
                start_time = self.utils.record_time()
                solution_syn_list = self.sequential_dse_pragma(selected_solutions, self.cfg.FPGA.clock_period)
            else:
                print('PYTHON : {} HLS threads are running in parallel. Starting at {}'.format(max_workers, dt_string))
                start_time = self.utils.record_time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    solution_syn_list = executor.map(self.parallel_dse_pragma, selected_solutions)
                solution_syn_list = list(solution_syn_list)

            self.copy_all_design_sourcefile()
            self.utils.save_a_variable('all_combined', solution_syn_list)
            #self.remove_all_synthesize_results(total_dse_solutions)
            [mm, ss] = self.utils.end_and_print_time(start_time)
            print("\nPYTHON : Total synthesis time : {:3d} Minutes and {:2d} Seconds".format(mm, ss))
        return solution_syn_list

    def run_dse_clock(self, options):
        '''
        Perform DSE on different frequencies at regular intervals within the specified clock range.
        :param options:
        :return:
        '''
        if options.mode == 'dse_clock_report':
            print("PYTHON : DSE_clock skipped...generating reports only")
            return '', ''
        else:
            # obtain a list of clock periods at regular intervals between min and max frequency
            clk_range = self.cfg.design_setting.DSE_setting['clock_range']
            freq_list = np.linspace(clk_range['min'],clk_range['max'],clk_range['samples']).astype(int)
            period_list = np.around(1000/freq_list,1)
            period_unit = 'ns'
            dse_clock_list = []
            sol = 0
            pragmas = self.hls_tools.create_fixed_directive_tcl_file(directive_type=self.cfg.design_setting.syn_directive_type)
            print("PYTHON : DSE on {} clock frequencies between {} and {} MHz !".format(len(period_list), clk_range['min'], clk_range['max']))

            # for each clk_period, run HLS synthesis and pickle results
            for clk_period in period_list:
                start_time = self.utils.record_time()
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                print("\nPYTHON : Running Clock DSE {}/{}: Synthesis of Design {} (period = {}{}) is started at {}.".format(
                    sol, len(period_list), sol, clk_period, period_unit, dt_string))
                self.hls_tools.create_syn_tcl_file(clk_period=clk_period)
                # run synthesis on design for this clk_period
                if self.hls_tools.run_hls_synth('syn', self.cfg.design_setting.syn_timeout, 'silent', clean=True, sol=sol):
                    beep('syn')
                    copy_solution_files(self.cfg, sol, specifier=clk_period)
                    self.hls_tools.copy_hls_bc_files(sol)
                    [mm, ss] = self.utils.end_and_print_time(start_time)
                    temp, model_layers_name = self.hls_tools.read_parallel_syn_results(sol, [mm,ss],False)
                    postSyn, power = self.hls_tools.run_vivado_implementation(sol, mode=options.mode, print_out='silent', clean=True)
                    temp.update(postSyn)
                    temp.update(power)
                    temp['dtype'] = '{} bits'.format(self.cfg.design_variable_types['ker_t'])
                    design = {'ID': clk_period,
                              'lyr_cfg': self.cfg.design_layers,
                              'syn_results': temp,
                              'analyzes': self.cfg.analyze_results,
                              'inserted_pragma': pragmas}
                    dse_clock_list.append(design)
                    self.utils.save_a_variable('design_{}'.format(clk_period), design)
                    self.utils.save_a_variable('dse_clock', dse_clock_list)
                    sol = sol + 1
        self.copy_all_design_sourcefile()
        return dse_clock_list

    def run_dse_pragma_clock(self, options):
        '''
        Perform DSE on different frequencies at regular intervals within the specified clock range.
        At each clock period, explore a set number of random pragma combinations.
        :param options:
        :return:
        '''
        if options.mode == 'dse_pragma_clock_report':
            print("PYTHON : DSE_pragma_clock skipped...generating reports only")
            return '', ''
        else:
            # obtain a list of clock periods at regular intervals between min and max frequency
            clk_range = self.cfg.design_setting.DSE_setting['clock_range']
            freq_list = np.linspace(clk_range['min'], clk_range['max'], clk_range['samples']).astype(int)
            period_list = np.around(1000 / freq_list, 1)
            print("PYTHON : DSE_pragma_clock on {} clock frequencies {} MHz !".format(len(freq_list), freq_list))

            # for each clk_period, run HLS pragma DSE and pickle results
            solution_syn_list = []
            [fixed_pragmas, variable_pragmas, total_comb] = self.hls_tools.pars_dnn_design_pragmas(self.cfg.pragmas['variable'])
            selection_type = self.cfg.design_setting.DSE_setting['directive_selection']
            selected_solutions = self.create_dse_directive_tcl_file(selection_type, fixed_pragmas, variable_pragmas, total_comb)
            num_solutions = len(selected_solutions)
            dse_count = 1
            total_dse_solutions = num_solutions*len(period_list)

            # for each clk_period, run HLS synthesis on every set of pragmas
            # pickle results in unique files
            for sol_num, sol in enumerate(selected_solutions):
                for clk_period in period_list:
                    syn_path = os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol),
                                            self.cfg.design_setting.solution_name)
                    start_time = self.utils.record_time()
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    print("\nPYTHON : DSE {}/{}: Synthesis of design {} (clk_period={}) is started at {}.".format(
                        dse_count, total_dse_solutions, sol, clk_period, dt_string))
                    self.hls_tools.create_syn_tcl_file(clk_period=clk_period)
                    design_name = '{}_{}'.format(sol, clk_period)
                    syn_status = self.hls_tools.run_hls_synth('syn', self.cfg.design_setting.syn_timeout, 'silent',
                                                              clean=True, sol=sol)
                    if not syn_status:
                        [mm, ss] = self.utils.end_and_print_time(start_time)
                        print(
                            "PYTHON : FAILED Synthesis DSE {}/{}: Synthesis of design {} is failed. Syn time :{:3d}:{:2d}".format(
                                dse_count, total_dse_solutions, sol, mm, ss))
                        beep('syn')
                        # if synth timesout, go to next set of pragmas
                        temp, model_layers_name = self.hls_tools.read_parallel_syn_results(sol, [mm, ss], False)
                        temp['solution'] = design_name
                        temp['dtype'] = '{} bits'.format(self.cfg.design_variable_types['ker_t'])
                        dse_count = dse_count + len(period_list)
                        break
                        time.sleep(2)
                        break
                    else:
                        beep('syn')
                        postSyn, power = self.hls_tools.run_vivado_implementation(sol, mode='dse_pragma',
                                                                                  print_out='silent', clean=True)
                        [mm, ss] = self.utils.end_and_print_time(start_time)
                        print(
                            "PYTHON : COMPLETED Synthesis DSE {}/{}: Synthesis of design {} is finished. Syn time :{:3d}:{:2d}".format(
                                dse_count, total_dse_solutions, sol, mm, ss))
                        temp, model_layers_name = self.hls_tools.read_parallel_syn_results(sol, [mm, ss], False)
                        temp['solution'] = design_name
                        temp['dtype'] = '{} bits'.format(self.cfg.design_variable_types['ker_t'])
                        temp.update(postSyn)
                        temp.update(power)
                        dse_count = dse_count + 1

                    design = {'ID': design_name,
                              'lyr_cfg': self.cfg.design_layers,
                              'syn_results': temp,
                              'analyzes': self.cfg.analyze_results,
                              'inserted_pragma': self.cfg.dse_pragmas[sol]}
                    copy_solution_files(self.cfg, sol, specifier=clk_period)
                    self.hls_tools.copy_hls_bc_files(sol, specifier=clk_period)
                    if self.cfg.design_setting.DSE_setting['remove_hls_run_directories'] and \
                            os.path.exists(os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol))):
                            shutil.rmtree(os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol)))
                    solution_syn_list.append(temp)
                    os.chdir(self.cfg.paths.design_top)
                    self.utils.save_a_variable('design{}'.format(design_name), design)
                    self.utils.save_a_variable('dse_pragma_clock', solution_syn_list)
        self.copy_all_design_sourcefile()
        return solution_syn_list

    def run_dse_dtype(self, design_name, options, variable_name):
        if options.mode == 'dse_dtype_report':
            print("PYTHON : dse_dtype_report Skipped")
            return '', ''
        else:
            dtype_list = self.cfg.design_setting.DSE_setting['dtype_range']
            dse_dtype_list = []
            sol=0
            given_var_types = self.cfg.design_variable_types
            self.hls_tools.create_fixed_directive_tcl_file(directive_type=self.cfg.design_setting.syn_directive_type)
            self.hls_tools.create_syn_tcl_file(clk_period=self.cfg.FPGA.clock_period)
            print("PYTHON : DSE on {} dtypes!".format(len(dtype_list[variable_name])))
            for precision in dtype_list[variable_name]:
                if variable_name in given_var_types:
                    given_var_types[variable_name] = precision
                else:
                    for dtype in ['in_t','ker_t','res_t']:
                        given_var_types[dtype] = precision
                    new_var_types = data_type_gen('ap_int',given_var_types)
                self.update_dtype_in_cfg_file(self.cfg.files.dnn_cfg_cppfile, new_var_types)

                start_time = self.utils.record_time()
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                print("\nPYTHON : DSE on dtype: Synthesis of design {} is started at {}.".format(sol,dt_string))
                if self.hls_tools.run_hls_synth('syn', self.cfg.design_setting.syn_timeout, 'silent' ,clean=True, sol=sol): # run a simple synthesize
                    beep('syn')
                    self.copy_solution_files(self.cfg.paths.solution,self.cfg.design_setting.DSE_setting['dse_name'], sol)
                    [mm, ss] = self.utils.end_and_print_time(start_time)
                    temp, model_layers_name = self.hls_tools.read_parallel_syn_results(sol, [mm,ss],False)
                    temp['dtype']='{} bits'.format(given_var_types['ker_t'])
                    postSyn, power = self.hls_tools.run_vivado_implementation(sol, mode=options.mode,print_out='silent', clean=True)
                    temp.update(postSyn)
                    temp.update(power)
                    dse_dtype_list.append(temp)
                    self.hls_tools.copy_hls_bc_files(sol)
                    if self.cfg.design_setting.DSE_setting['remove_hls_run_directories'] and \
                        os.path.exists(os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol))):
                        shutil.rmtree(os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol)))
                    self.utils.save_a_variable('design_{}'.format(sol), temp)
                    self.utils.save_a_variable('dse_dtype', dse_dtype_list)
                sol = sol + 1
        self.copy_all_design_sourcefile()
        t = list(dse_dtype_list[0].keys())
        model_layers_name = t[0:t.index(self.cfg.design_setting.topmodule)]
        return dse_dtype_list, model_layers_name

    def run_dse_variable_change(self,NewValue_dict, options):
        if options.mode == 'dse_variable_report':
            print("PYTHON : DSE Skipped")
            dse_var_list = self.utils.load_a_variable('dse_variable')
        else:
            targetFile = 'C:/Users/mshahshahani/Documents/cnn_research/dnn_small/lightFC/lightFC.h'
            dse_var_list = []
            sol = 0
            self.hls_tools.create_fixed_directive_tcl_file(directive_type=self.cfg.design_setting.syn_directive_type)
            self.hls_tools.create_syn_tcl_file(clk_period=self.cfg.FPGA.clock_period)
            print('PYTHON : DSE on variable {} is running ... '.format(NewValue_dict.keys()))
            for VAR in NewValue_dict.keys():
                for VAL in NewValue_dict[VAR]:

                    self.replace_a_variable_value_in_file(targetFile, VAR, VAL)

                    start_time = self.utils.record_time()
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    print("\nPYTHON : DSE on variable {}={}: Synthesis of design {} is started at {}.".format(VAR, VAL,
                                                                                                              sol,
                                                                                                              dt_string))
                    if self.hls_tools.run_hls_synth('syn', self.cfg.design_setting.syn_timeout,'silent', clean=True, sol=sol):
                        beep('syn')
                        self.copy_solution_files(self.cfg.paths.solution, self.cfg.design_setting.DSE_setting['dse_name'], sol)
                        self.hls_tools.copy_hls_bc_files(sol)
                        [mm, ss] = self.utils.end_and_print_time(start_time)
                        temp, model_layers_name = self.hls_tools.read_single_syn_results(sol, [mm, ss], False)
                        temp['dtype'] = '{} bits'.format(self.cfg.design_variable_types['ker_t'])
                        postSyn, power = self.hls_tools.run_vivado_implementation(sol, mode=options.mode, print_out='silent', clean=True)
                        temp.update(postSyn)
                        temp.update(power)
                        dse_var_list.append(temp)
                        self.hls_tools.copy_hls_bc_files(sol)
                        if self.cfg.design_setting.DSE_setting['remove_hls_run_directories'] and \
                                os.path.exists(os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol))):
                            shutil.rmtree(os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol)))
                        self.utils.save_a_variable('dse_variable', dse_var_list)
                    else:
                        exit()
                    sol = sol + 1
        self.copy_all_design_sourcefile()
        return dse_var_list

