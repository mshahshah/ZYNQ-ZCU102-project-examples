# ////////////////////////////////////////////////////////////////////////////////////////
# Copyright (c) by
# Company:  IDEA LAB, The University of Texas at Dallas
# Author :  Masoud Shahshahani
#
# Originally Create Date: Mar-5, 2020
# Project Name    : DNN_Framework
# Tool Versions   : Python +3
#
# Description     : Main file for executing submodules and functions based on the input arguments
# Dependencies    :
# Additional Comments:

# Primary python code for implementing DNN ACE tool, a framework for generating FPGA implementations of DNNs
# Includes scripts for calling Vitis HLS and Vivado synthesis and implementation.
# Also has to ability to perform Design Space Exploration to find pareto-optimum design implementations
#
# ///////////////////////////////////////////////////////////////////////////////////////

import os,sys, copy,numpy, yaml, shutil
import argparse, random, glob,pandas, pickle
import os, sys, copy, yaml
import argparse, random, glob, pandas, pickle
import numpy as np
print("Welcome! dnn_framework.py is executed")
cur_path = os.getcwd()
sys.path.append(cur_path)
sys.path.append(os.path.join(cur_path, 'src'))
sys.path.append(os.path.join(cur_path, 'sim'))
sys.path.append(os.path.join(cur_path, 'src/framework'))

from utils import *
from utils import beep
from dnn_tools import *
from hls_tools import *
from dnn_analyzer import *
from dnn_DSE import *
from cnn_net_builder import cnn_net_builder
from hls_tools import collect_design_notes
from PPA_estimator import *

lyr_map2syn = {
    'LeNet':'dnn_LeNet',
    'C1P1F1':'dnn_C1P1F1',
    'C2P2F1':'dnn_C2P2F1',
    'AlexNet':'dnn_AlexNet',
    'ConvNet':'dnn_ConvNet',
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


def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-mode"   , help="To run synthesize." , default='skip')
    parser.add_argument("-clean"  , help="To clean before synthesize" , default='False')
    #parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
    options = parser.parse_args(args)
    return options



# ###################################################################
# ######################  ALL  DSE    ###############################

def run_dse(cfg, dse_modules):
    '''
    Performs Design Space Exploration (DSE) for the type of exploration specified by the python run options
    :param cfg: configuration object
    :param dse_modules: Not used
    :return: none
    '''
    lyr_configs = []
    if cfg.run_options.mode in ['dse_cfg', 'dse_cfg_report']:
        run_dse_cfg(utils, hls_tools, dnn_analyzer, dnn_tools, cfg)
        design_solutions = dnn_dse.create_dse_excel_report_new(target_layers='all', add_cfg_labels=True)
        path = cfg.paths.dse_report
        create_data_for_estimation(dnn_dse, dnn_analyzer, cfg, path)
    elif cfg.run_options.mode in ['dse_pragma', 'dse_pragma_report']:
        design_solutions = dnn_dse.run_dse_pragma(cfg.design_setting.topmodule, cfg.run_options)
        design_solutions = dnn_dse.create_dse_excel_report_new(target_layers='all', add_cfg_labels=False)
    elif cfg.run_options.mode in ['dse_clock', 'dse_clock_report']:
        design_solutions = dnn_dse.run_dse_clock(cfg.run_options)
        design_solutions = dnn_dse.create_dse_excel_report_new(target_layers='all', add_cfg_labels=False)
        #dnn_dse.create_dse_excel_report(design_solutions, layers=model_layers_name, lyr_configs=lyr_configs,
        #                                sort1_by='exec us', sort2_by='DSP')
    elif cfg.run_options.mode in ['dse_pragma_clock', 'dse_pragma_clock_report']:
        design_solutions = dnn_dse.run_dse_pragma_clock(cfg.run_options)
        design_solutions = dnn_dse.create_dse_excel_report_new(target_layers='all', add_cfg_labels=False)
    elif cfg.run_options.mode in ['dse_clk_pragma_cfg', 'dse_clk_pragma_cfg_report']:
        run_dse_clk_pragma_cfg(cfg)
        design_solutions = dnn_dse.create_dse_excel_report_new(target_layers='all', add_cfg_labels=True)
        path = cfg.paths.dse_report
        create_data_for_estimation(dnn_dse, dnn_analyzer, cfg, path)
    elif cfg.run_options.mode in ['dse_dtype', 'dse_dtype_report']:
        design_solutions, model_layers_name = dnn_dse.run_dse_dtype(cfg.design_setting.topmodule, cfg.run_options,'all_variables')
        dnn_dse.create_dse_excel_report(design_solutions, layers=model_layers_name, lyr_configs=lyr_configs,
                                        sort1_by='exec us', sort2_by='DSP')
    elif cfg.run_options.mode in ['dse_universal', 'dse_universal_report']:
        design_solutions, model_layers_name = dnn_dse.run_dse_universal(cfg.design_setting.topmodule, cfg.run_options)
        dnn_dse.create_dse_excel_report(design_solutions, layers=model_layers_name, lyr_configs=lyr_configs,
                                        sort1_by='exec us', sort2_by='DSP')
    elif cfg.run_options.mode in ['dse_variable', 'dse_variable_report']:
        NewValue_dict = {'M': [5, 10, 15, 20]}
        design_solutions = dnn_dse.run_dse_variable_change(NewValue_dict, cfg.run_options)
        dnn_dse.create_dse_excel_report(design_solutions, layers=model_layers_name, lyr_configs=lyr_configs,
                                        sort1_by='exec us', sort2_by='DSP')
    else:
        print("PYTHON : Please enter a correct DSE type !")
        exit()

    dnn_dse.copy_all_json_files()
    shutil.make_archive(cfg.paths.dse_report, 'zip', cfg.paths.dse_report)

    #dnn_dse.plot_dse(design_solutions, dse_modules, True)

    beep('finish')

# ###################################################################
# ######################  Syn & Iml   ###############################

def run_syn(cfg, print_out):
    '''
    Runs HLS synthesis using hls_tools under the constraints specified in cfg
    :param cfg: configuration object
    :param print_out: 'silent' for no print outs
    :return: (syn_rslt_summary, read_single_syn_results)
    '''
    start_time = utils.record_time()
    hls_tools.create_fixed_directive_tcl_file(directive_type=cfg.design_setting.syn_directive_type)
    hls_tools.create_syn_tcl_file(clk_period=cfg.FPGA.clock_period)
    if (not hls_tools.run_hls_synth(mode=options.mode, time_out_min=cfg.design_setting.syn_timeout,
                                    print_out=print_out, clean=options.clean)):
        sys.exit("PYTHON : Stoped Prcoess because of above errors\n")
    else:
        [min, sec] = utils.end_and_print_time(start_time)
        syn_rslt_summary, read_single_syn_results = hls_tools.read_single_syn_results(1, [min, sec], print_out=True)
    beep('syn')
    return syn_rslt_summary, read_single_syn_results

def run_sim(cfg, print_out):
    '''
    Performs the simulation only
    :param cfg: configuration object
    :param print_out: 'silent' for no print out
    :return: none
    '''
    start_time = utils.record_time()
    hls_tools.create_fixed_directive_tcl_file(directive_type=cfg.design_setting.syn_directive_type)
    hls_tools.create_syn_tcl_file(clk_period=cfg.FPGA.clock_period)
    if (not hls_tools.run_hls_synth(mode=options.mode, time_out_min=cfg.design_setting.syn_timeout,
                                    print_out=print_out, clean=options.clean)):
        sys.exit("PYTHON : Stoped Prcoess because of above errors\n")
    else:
        [min, sec] = utils.end_and_print_time(start_time)
    return

def run_impl(cfg, print_out):
    '''
    Performs FPGA implementation in Vivado using the supplied configuration.
    :param cfg: configuration object
    :param print_out: 'silent' for no print out
    :return: postSyn, power
    '''
    start_time = utils.record_time()
    postSyn, power = hls_tools.run_vivado_implementation(0, mode=options.mode, print_out=print_out, clean=True)
    [min, sec] = utils.end_and_print_time(start_time)
    print('Total On-Chip Power : {} (mW) '.format(power['P_Total']))
    beep('impl')
    return postSyn, power





##############################################################################
###########################   MAIN   #########################################
##############################################################################

if __name__ == '__main__':
    '''    
    Some possible options to execute dnn_framework.py:
    To run synthesize, clean solutions, show results:
           python dnn_framework.py -mode syn -clean True
           python dnn_framework.py -mode syn
    To Skip synthesize, show results:
           python dnn_framework.py -mode skip -clean True
           python dnn_framework.py
    To Skip synthesize and DSE, like breaking the DSE before it finishes:
           python dnn_framework.py -mode report -clean False
        To Skip synthesize and run DSE :
           python dnn_framework.py -mode dse -dse_name dse2 -clean False
    '''
    os.system('clear')

    start_time = time.time()

    # Parse the input paramaters from input_arguments.yaml
    options = getOptions()

    gen_configs = configure_design()
    cfg = gen_configs.create_cfg(options)
    gen_configs.prepare_design()

    # Initialize dnn and hls objects
    utils = utils(cfg)
    dnn_tools = dnn_tools(cfg)
    hls_tools = hls_tools(cfg)
    dnn_dse = dnn_dse(cfg)

    # Parse the specified dnn yaml file
    given_lyrs, given_var_types = dnn_tools.pars_user_layer_defined()
    dnn_tools.print_user_layer_defined(given_lyrs)

    utils.end_and_print_time_and_label(start_time, 'Parsing User data')
    start_time = utils.record_time()
    # -----------------------------------------------------------------------------------------------------
    predicted_label = [0,0]
    if cfg.design_setting.training_setting['train_model']:
        cnn_net_builder = cnn_net_builder(cfg)
        if cfg.design_setting.design_model.lower() == 'lenet':
            lyrs=[{"fi":1, "fo":7, "k":5, "s":1},
                  {"fi":7, "fo":16, "k":5, "s":1},
                  {"fi":16*5*5, "fo":120, "k":1, "s":1},
                  {"fi":120, "fo":84, "k":1, "s":1},
                  {"fi":84, "fo":10, "k":1, "s":1}]
            model = cnn_net_builder.selec_model('lenet', given_lyrs)
            train_loader, test_loader = cnn_net_builder.load_data()
            cnn_net_builder.save_test_images(test_loader)
            model = cnn_net_builder.train_model(model, train_loader, test_loader)
            cnn_net_builder.test_model(model, test_loader)
            cnn_net_builder.export_trained_weights(model)
            predicted_label = cnn_net_builder.test1pic(model, test_loader, sample=cfg.design_setting.training_setting['test_sample_num'])
            predicted_label = predicted_label.mul(200).int().tolist()[0]
            cnn_net_builder.create_training_testing_report()
        utils.end_and_print_time_and_label(start_time, 'Training and testing')
        start_time = utils.record_time()

    dse_modules = ['conv_3DT1', 'ds_3DT1', 'fc_T1', 'conv2fc', 'dnn_LeNet']
    # -----------------------------------------------------------------------------------------------------
    if cfg.design_setting.rebuil_hls:
        dnn_configs = dnn_tools.create_dnn_configs_file(given_lyrs, given_var_types)
        cpp_segments = dnn_tools.create_main_cpp_file(dnn_configs, predicted_label)
        dnn_tools.create_main_header_file(cpp_segments)
        # modify_testbench so that the UUT is declared correctly
        utils.replace_word_in_file(os.path.join(cfg.paths.design_model, 'top_tb.cpp'), 'UUT', cfg.design_setting.topmodule)
        #dnn_tools.read_user_layer_defined()

        utils.end_and_print_time_and_label(start_time, 'Code generation')
        start_time = utils.record_time()
    # -----------------------------------------------------------------------------------------------------
    if cfg.design_setting.quantitative_analysis:
        dnn_analyzer = dnn_analyzer(cfg)
        analyze_results = dnn_analyzer.analyze_given_model(given_lyrs)
        dnn_analyzer.overall_report(analyze_results, log=cfg.design_setting.analysis_log)
        dnn_analyzer.save_overall_report(fname='analyzes', plot=False)
        utils.end_and_print_time_and_label(start_time, 'Analyzing')
        PPA_estimator = PPA_estimator(cfg, hls_tools, dnn_tools, dnn_analyzer)
        dnnEstimator = PPA_estimator.train_model_and_estimate(cfg, given_lyrs)

    # -----------------------------------------------------------------------------------------------------
    if 'dse' in options.mode:
        run_dse(cfg, dse_modules)
        utils.end_and_print_time_and_label(start_time, 'DSE finished')
        start_time = utils.record_time()

    elif options.mode in ['sim', 'sim_report', 'all']:
        start_time = utils.record_time()
        run_sim(cfg, 'silent')
        [mm, ss] = utils.end_and_print_time_and_label(start_time,
                   'Simulation is completed {}'.format(cfg.design_setting.topmodule))

    elif options.mode in ['syn', 'syn_report', 'all']:
        start_time = utils.record_time()
        syn_rslt_summary = run_syn(cfg, 'silent')
        dnnEstimator.results_deviation(dnnEstimator.top_estimation, syn_rslt_summary, print_out=True)
        dnn_analyzer.results_deviation(syn_rslt_summary, print_out=True)
        postSyn, power = run_impl(cfg, '')
        [mm,ss] = utils.end_and_print_time_and_label(start_time, 'Syn is completed {}'.format(cfg.design_setting.topmodule))
        hls_tools.save_syn_records(syn_rslt_summary, postSyn, [mm,ss])


    elif options.mode in ['evaluate_model', 'evaluate_model_report']:
        design_lists = [0,1,2,3,4,5,6,7,8,9,10]
        #design_lists = [0,3,7,9,10]
        design_names = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E',
                        5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K'}
        evaluated_models = evaluate_estimation_model(dnnEstimator, design_lists)
        solutions_list = dnn_dse.create_dse_excel_report_new(target_layers='top')
        new_design_names = {}
        for i in design_lists:
            new_design_names[i] = design_names[i]
        dnn_dse.plot_evaluated_models(new_design_names, solutions_list)
    else:
        print("PYTHON : Syn Skipped")
        print("PYTHON : DSE Skipped")

    if options.mode in ['syn', 'map_hw', 'all']:
        start_time = utils.record_time()
        hls_tools.prepare_hw_modules()
        vivado_results = hls_tools.createNrun_hw_architecture()
        if not vivado_results == False:
            [mm,ss] = utils.end_and_print_time_and_label(start_time, 'The hardware is mapped in Vivado ')
        else:
            print("PYTHON : Mapping Skipped")


print('PYTHON : Task Completed !')






# precision = 16
#
# example_float = [0.5, 0.3, 0.99, -0.2, -.00024, 0.0043]
# print('Floating point: ', example_float)
#
# example_fixed = dnn_tools.float2int(example_float, precision, 'ceil')
# print('Fixed point:', example_fixed)
#
# example_hex = dnn_tools.dec2hex(example_fixed,precision)
# print('HEX:        ', example_hex)
#
# example_dec2 = dnn_tools.hex2dec(example_hex,precision)
# print('dec :       ', example_dec2)
#
# dnn_tools.create_coe_file('test_dec',example_fixed,'dec',reshape=1)
# dnn_tools.create_coe_file('test_hex',example_hex,'hex',reshape=1)
