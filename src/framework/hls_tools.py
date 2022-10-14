# ////////////////////////////////////////////////////////////////////////////////////////
# Copyright (c) by
# Company:  IDEA LAB, The University of Texas at Dallas
# Author :  Masoud Shahshahani
#
# Originally Create Date: Mar-5, 2020
# Project Name    : DNN_Framework
# Tool Versions   : Python +3
#
# Description     : Functions to run HLS and Logic synthesize, create .tcl files, load design reports
#                   Pars pragmas and create .tcl file for directives
# Dependencies    : Vivado 2018 or newer, subprocess, panda
# Additional Comments:
#
# ///////////////////////////////////////////////////////////////////////////////////////

import os, shutil
import csv, time
from datetime import datetime
from shutil import copyfile
import sys, glob, json, random, pickle
import numpy as np
import yaml, random
import subprocess, psutil
from dnn_tools import *
from utils import beep
import pandas as pd
import pprint
import xml.etree.ElementTree as ET



lyr_map2syn = {
    'IN':'read_input3D',
    'CONV':'conv_3DT1',
    'POOL':'ds_3DT1',
    'FC':'fc_T1'}

lyr_syn2map = {
    'read_input3D':'IN',
    'conv_3DT1':'CONV',
    'ds_3DT1':'POOL',
    'fc_T1':'FC'}


def copy_solution_files(cfg, sol, specifier=None):
    if specifier == None :  temp = ''
    else:          temp = '_' + str(specifier)
    sol_copy_name = os.path.join(cfg.paths.dse_json, 'design' + str(sol) + temp + '.json')
    json_file = os.path.join(cfg.paths.design_model, 'hls' + str(sol), cfg.design_setting.solution_name,
                             '{}_data.json'.format(cfg.design_setting.solution_name))
    if os.path.exists(json_file):
        copyfile(json_file, sol_copy_name)
    else:
        print("PYTHON : Copy Solution Files : Solution file is not generated")

def collect_design_notes(cfg, additional_info, save_path=''):
    notes = []
    notes.append('Time completed   : {}'.format(datetime.now().strftime("%d-%b-%Y (%H:%M)")))
    notes.append('DSE Path         : {}'.format(os.getcwd()))
    notes.append('\n')
    notes.append('Design           : {}'.format(cfg.design_setting.topmodule))
    notes.append('FPGA             : {}'.format(cfg.FPGA.chip))
    notes.append('Clock            : {}'.format(cfg.FPGA.clock_freq))
    notes.append('Vivado Version   : {}'.format(cfg.design_setting.vivado_version))
    notes.append('Logic Synthesize : {}'.format(cfg.design_setting.run_vivado_synthesize))
    notes.append('\n')
    notes.append('Directive_type   : {}'.format(cfg.design_setting.syn_directive_type))
    notes.append('Dataflow         : {}'.format(cfg.design.dataflow))
    notes.append('Shared Memory    : {}'.format(cfg.design.fc_shared_memory))
    notes.append('data_interface   : {}'.format(cfg.design.data_interface))
    notes.append('Module Interface : {}'.format(cfg.design.module_controller))
    notes.append('\n')
    for i in additional_info.keys():
        notes.append('{} : {}\n'.format(i, additional_info[i]))

    if 'dse' in cfg.run_options.mode:
        for i in cfg.design_setting.DSE_setting.keys():
            notes.append('{:<30s} : {}'.format(i, cfg.design_setting.DSE_setting[i]))
    file = os.path.join(save_path, 'design note.txt')
    with open(file, 'w') as f:
        for line in notes:
            f.write("%s\n" % line)
    return notes


def extract_layer_info_from_jsons(cfg, json_path):
    path = cfg.paths.dse_report
    dse_results = utils.load_a_variable(os.path.join(path,'dse_config'))
    cfg_results = utils.load_a_variable(os.path.join(path, 'lyr_configs'))
    module_labels = ['clock period', 'latency', 'BRAM', 'LUT', 'FF', 'DSP']
    sol_labels = ['syn_time','P_slice', 'P_BRAM', 'P_DSP', 'P_static', 'P_total','LUT_PS','FF_PS','DSP_PS','BRAM_PS']
    lyr_label = ['w_in','w_out','lyr_in','lyr_out','w_ker','stride']
    target_layer_name ='fc_T1'
    target_layer = 3
    rpt_data = []
    for sol in range(len(dse_results)):
        temp = []
        for label in sol_labels:
            temp.append(dse_results[sol][label])
        for label in lyr_label:
            temp.append(cfg_results[sol][target_layer][label])
        for label in module_labels:
            temp.append(dse_results[sol][target_layer_name][label])
        rpt_data.append(temp)

    header_labels = sol_labels + lyr_label + module_labels
    csvfile = os.path.join(path,'dse_cfg_'+target_layer_name+'.csv')
    df = pandas.DataFrame(rpt_data)
    df.to_csv(csvfile, index=False, header=header_labels)



class hls_tools():
    def __init__(self, cfg):
        self.cfg = cfg
        self.utils = utils(cfg)
        self.pragmas_dict = {
            'unroll': 'set_directive_unroll         ',
            'pipeline': 'set_directive_pipeline       ',
            'dataflow': 'set_directive_dataflow       ',
            'inline': 'set_directive_inline         ',
            'partition': 'set_directive_array_partition',
            'reshape': 'set_directive_array_reshape  ',
            'interface': 'set_directive_interface      ',
            'mul': 'set_directive_allocation     '
        }

    def create_syn_tcl_file(self, clk_period):
        tcl_lines = []
        tcl_lines.append("############################################################")
        tcl_lines.append("## This file is generated automatically by python tool for {} Version".format(
            self.cfg.design_setting.vivado_version))
        tcl_lines.append("############################################################")
        tcl_lines.append('puts \"CMD : run_hls_syn.tcl is running!\"')
        tcl_lines.append('set sol_name [lindex  $argv 2 ]')
        tcl_lines.append('open_project hls$sol_name')
        tcl_lines.append('set_top   {}'.format(self.cfg.design_setting.topmodule))
        for file in self.cfg.design.source_files:
            tcl_lines.append('add_files  {}'.format(file))

        for file in self.cfg.design.tb_files:
            tcl_lines.append('add_files  -tb {}'.format(file))
        if self.cfg.design_setting.vivado_version == 2020:
            tcl_lines.append('open_solution -reset \"{}\"  -flow_target vivado'.format(self.cfg.design_setting.solution_name))
            tcl_lines.append('set_part {' + self.cfg.FPGA.part + '}')
        else:
            tcl_lines.append('open_solution -reset \"{}\"'.format(self.cfg.design_setting.solution_name))
            tcl_lines.append('set_part {' + self.cfg.FPGA.part + '} -tool vivado')
        tcl_lines.append('create_clock -period {} -name default'.format(clk_period))
        tcl_lines.append('set_clock_uncertainty 12.5%')
        tcl_lines.append("set json_file \"{}_{}.json\"".format(self.cfg.design_setting.solution_name, clk_period))
        if 'dse_pragma' in self.cfg.run_options.mode:
            tcl_lines.append('source \"{}_sol_list/solution_$sol_name.tcl\"'.format(self.cfg.design_setting.solution_name))
        else:
            tcl_lines.append('source \"./hls/{}/directives.tcl\"'.format(self.cfg.design_setting.solution_name))
        if self.cfg.design_setting.Sim_setting['run_csim']:
            tcl_lines.append('csim_design')

        if self.cfg.run_options.mode not in ['sim'] or self.cfg.design_setting.Sim_setting['run_rtl_sim']:
            tcl_lines.append('csynth_design')

        if self.cfg.design_setting.Sim_setting['run_rtl_sim']:
            tcl_lines.append('cosim_design')

        if self.cfg.design_setting.run_vivado_synthesize:
            tcl_lines.append('export_design -flow syn -rtl verilog -format ip_catalog')
        elif self.cfg.design_setting.create_ip:
            tcl_lines.append('export_design -format ip_catalog')

        tcl_lines.append('quit')
        filename = os.path.join(self.cfg.paths.design_model, "run_hls_syn.tcl")
        self.utils.save_list_to_file(filename, tcl_lines)

    def run_vivado_implementation(self, sol_counter, mode, print_out='silent', clean=False):
        if not self.cfg.design_setting.run_vivado_power_analyzer:
            PR_results = {'LUT_PS': 'NR', 'FF_PS': 'NR', 'DSP_PS': 'NR', 'BRAM_PS': 'NR', 'Timing_PS': 'NR'}
            power = {'P_Clocks': 'NR', 'P_Signals': 'NR', 'P_Slice': 'NR', 'P_Block': 'NR', 'P_DSPs': 'NR', 'P_Static': 'NR', 'P_Total': 'NR'}
            return PR_results, power

        if self.cfg.run_options.mode in ['dse_pragma', 'dse_pragma_clock', 'dse_clk_pragma_cfg']:
            dest_path = os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol_counter),
                                     self.cfg.design_setting.solution_name)
        else:
            dest_path = self.cfg.paths.solution

        start_time = self.utils.record_time()
        impl_file = ['power_analyzer.tcl', 'run_power_analyzer.bat']
        for fname in impl_file:
            srcfile = os.path.join(self.cfg.paths.src, fname)
            destfile = os.path.join(dest_path, 'impl', fname)
            shutil.copyfile(srcfile, destfile)
        os.chdir(os.path.join(dest_path, 'impl'))
        
        
        version = self.cfg.design_setting.vivado_version
        vivado_cmd = self.cfg.vivado_path[sys.platform][version]['VIVADO']
        print("PYTHON : Running Power Analyzer ... ")
        if sys.platform == 'win32':

            sr = os.system('run_power_analyzer.bat')
        elif sys.platform == 'linux':
            cmd = '{}  -notrace -mode batch -source power_analyzer.tcl >report_power.log || exit $?'.format(vivado_cmd)
            sr = os.system(cmd)
        else:
            print("PYTHON : Wrong operating system selection")
            sr = 1

        if (sr != 0):
            print("PYTHON : run_power_analyzer file not found, or a problem in bash file!")
            return 'Er'

        [mm, ss] = self.utils.end_and_print_time(start_time)

        os.chdir(self.cfg.paths.design_top)
        print("PYTHON : Power measured in Vivado  within {:3d} Minutes and {:2d} Seconds".format(int(mm), int(ss)))
        return self.read_impl_results(dest_path)

    def cleanSolution(self):
        filesToBeRemoved = [self.cfg.files.synLogFile, self.cfg.files.SolutionFile]
        dir_list = [self.cfg.paths.rtl_out, self.cfg.paths.hls_out]
        for fname in filesToBeRemoved:
            if os.path.exists(fname):
                os.remove(fname)
                os.mkdir(fname)
            else:
                os.mkdir(fname)
        for dir in dir_list:
            if os.path.exists(dir):
                shutil.rmtree(dir)
                os.mkdir(dir)
            else:
                os.mkdir(dir)

    def run_hls_synth(self, mode, time_out_min=10, print_out='silent', clean=False, sol=''):
        if mode in ['', 'skip', 'syn_report', 'dse_report']:
            print("PYTHON : Synthesize Skipped\n")
            return True
        version = self.cfg.design_setting.vivado_version
        hls_cmd = self.cfg.vivado_path[sys.platform][version]['HLS']
        os.chdir(self.cfg.paths.design_model)

        print("PYTHON : Synthesis begins from python on {} Version {}".format(sys.platform, version))
        if print_out == 'silent':
            cmd = "{} -f run_hls_syn.tcl {} > synthesis_report{}.log".format(hls_cmd, sol, sol)
        else:
            cmd = "{} -f run_hls_syn.tcl {}".format(hls_cmd, sol)
        if clean:
            self.cleanSolution()

        start_time = self.utils.record_time()

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        try:
            if not print_out == 'silent': print("The task PID is = {}".format(p.pid))
            p.wait(time_out_min*60)
            sr = 0
        except subprocess.TimeoutExpired:
            print("WARNING Timeout : process {} is killed by reaching {} Min!".format(p.pid, time_out_min))
            sr = 1
            for child in psutil.Process(p.pid).children(recursive=True):
                if not print_out == 'silent': print("PID={:<4}, name={} is Killed".format(child.pid, child.name()))
                child.kill()
            p.kill()
            time.sleep(1)

        os.chdir(self.cfg.paths.design_top)
        [mm, ss] = self.utils.end_and_print_time(start_time)

        if (sr != 0):
            print("PYTHON : Synthesis file not found, or a problem in bash file!")
            return False
        self.cfg.files.synLogFile = os.path.join(self.cfg.paths.design_model,'synthesis_report{}.log'.format(sol))
        errors = self.utils.find_Aword_in_file(self.cfg.files.synLogFile, 'error', save_results=False)
        # warning = self.utils.find_Aword_in_file(self.cfg.files.synLogFile, 'warning', save_results=True)
        if errors != 0:
            print("PYTHON : *** Synthesize Failed ***  - Total synthesis time : {:3d} Minutes and {:2d} Seconds".format(
                int(mm), int(ss)))
            copyfile(self.cfg.files.synLogFile,
                     os.path.join(self.cfg.paths.solution, "failed_syn_log_sol{}.log".format(sol)))
            return False
        else:
            print("PYTHON : *** Synthesize Passed ***  - Total synthesis time : {:3d} Minutes and {:2d} Seconds".format(
                int(mm), int(ss)))
            return True



    def read_single_syn_results_unused(self, print_out=False):
        solutions_list = []
        file = self.cfg.paths.solution + "/ip_test_data.json"
        with open(file) as json_data:
            syn_rslt = json.load(json_data)
            json_data.close()
            # except IOError:
            #    print("I/O Error," + " solution"+ str(solution) + "_Syn.json "+ " is not exist")

        temp = syn_rslt["ModuleInfo"]["Metrics"][self.design.top_module]
        syn_rslt_summary = {}
        syn_rslt_summary['latencyBest'] = int(temp["Latency"]["LatencyBest"])
        syn_rslt_summary['latencyWrst'] = int(temp["Latency"]["LatencyWorst"])
        syn_rslt_summary['timing'] = float(temp["Timing"]["Estimate"])
        syn_rslt_summary['Target'] = float(temp["Timing"]["Target"])

        syn_rslt_summary['BRAM'] = int(temp["Area"]["BRAM_18K"])
        syn_rslt_summary['LUT'] = int(temp["Area"]["LUT"])
        syn_rslt_summary['FF'] = int(temp["Area"]["FF"])
        syn_rslt_summary['DSP'] = int(temp["Area"]["DSP48E"])

        if print_out:
            print(json.dumps(syn_rslt_summary, indent=6))
        return syn_rslt_summary

    def single_pragma_gen(self, target_module_label, configs):
        options = ''
        if configs['options'] is not None:
            for i in configs['options']:
                options = options + "-{}  ".format(i)
        if configs['pragma'] in ['unroll', 'pipeline', 'inline']:
            directive = "{}  {} \"{}/{}\"".format(self.pragmas_dict[configs['pragma']], options, target_module_label,
                                                  configs['label'])
        elif configs['pragma'] in ['partition', 'reshape', 'interface']:
            directive = "{}  {} \"{}\" {}".format(self.pragmas_dict[configs['pragma']], options, target_module_label,
                                                  configs['label'])
        elif configs['pragma'] in ['mul']:
            directive = "{}  {} \"{}/{}\" {}".format(self.pragmas_dict[configs['pragma']], options, target_module_label,
                                                     configs['label'], configs['pragma'])
        # print(directive)
        return directive

    def combinational_pragma_gen(self, target_module_label, configs):
        options_list = []
        if configs['options'] is None:  # if there is no option, create an empty one
            options_list.append('')
        else:  # if there is a option, add a no option
            # options_list.append('')
            for option in configs['options']:  # create directives for all options
                options_list.append('-' + option)

        directives_list = []
        directives_list.append('')  # create an empty directive
        for option in options_list:
            if configs['pragma'] in ['unroll', 'pipeline', 'dataflow']:
                directive = "{}  {}   \"{}/{}\"".format(self.pragmas_dict[configs['pragma']], option,
                                                        target_module_label, configs['label'])
            elif configs['pragma'] in ['partition', 'reshape', 'interface']:
                directive = "{}  {}   \"{}\" {}".format(self.pragmas_dict[configs['pragma']], option,
                                                        target_module_label, configs['label'])
            elif configs['pragma'] in ['mul']:
                directive = "{}  {}   \"{}/{}\" {}".format(self.pragmas_dict[configs['pragma']], option,
                                                           target_module_label, configs['label'], configs['pragma'])
            elif configs['pragma'] in ['inline']:
                directive = "{}  {}   \"{}\"".format(self.pragmas_dict[configs['pragma']], option,
                                                     target_module_label)
            directives_list.append(directive)

        return directives_list

    def find_total_pragma_combinations(self, variable_pragmas):
        total_comb = 1
        for pragma in variable_pragmas:
            total_comb = total_comb * len(pragma)
        return total_comb

    def pars_DA_design_pragmas(self, datamap):
        fixed_pragmas = []
        variable_pragmas = []
        # fixed_pragmas.append('\n# --------------------------  {} ----------------------------'.format(key))
        # fixed_pragmas.append('# ----------------------------------------------------------------')
        for module in datamap.keys():
            fixed_pragmas.append('')
            modules = datamap[module]
            if (datamap[module]) is not None:
                for zone in modules.keys():
                    if datamap[module][zone] is not None:
                        for pragma in datamap[module][zone]:
                            if module == 'top_module':
                                target_module_label = self.cfg.design_setting.topmodule
                            else:
                                target_module_label = module
                            temp = self.combinational_pragma_gen(target_module_label, pragma)

                            if pragma['type'] == 'fix':
                                fixed_pragmas.append(temp[1])
                            elif pragma['type'] == 'var':
                                variable_pragmas.append(temp)

        total_comb = self.find_total_pragma_combinations(variable_pragmas)
        return fixed_pragmas, variable_pragmas, total_comb

    def pars_design_pragmas(self, datamap):
        fixed_pragmas = []
        variable_pragmas = []
        # fixed_pragmas.append('\n# --------------------------  {} ----------------------------'.format(key))
        # fixed_pragmas.append('# ----------------------------------------------------------------')
        for module in datamap.keys():
            fixed_pragmas.append('')
            modules = datamap[module]
            if (datamap[module]) is not None:
                for zone in modules.keys():
                    if datamap[module][zone] is not None:
                        for pragma in datamap[module][zone]:
                            if module == 'top_module':
                                target_module_label = self.cfg.design_setting.design_model
                            else:
                                target_module_label = module
                            temp = self.combinational_pragma_gen(target_module_label, pragma)

                            if pragma['type'] == 'fix':
                                fixed_pragmas.append(temp[1])
                            elif pragma['type'] == 'var':
                                variable_pragmas.append(temp)

        total_comb = self.find_total_pragma_combinations(variable_pragmas)
        return fixed_pragmas, variable_pragmas, total_comb

    def pars_dnn_design_pragmas(self, datamap):
        fixed_pragmas = []
        variable_pragmas = []
        for key in datamap.keys():
            intf = datamap[key]
            # fixed_pragmas.append('\n# --------------------------  {} ----------------------------'.format(key))
            # fixed_pragmas.append('# ----------------------------------------------------------------')
            for module in intf.keys():
                modules = intf[module]
                if (intf[module]) is not None:
                    parse_zone = intf[module].get('parse', False)
                    for zone in ['array', 'function', 'loop', 'resources']:
                        if intf[module].get(zone, None) is not None:
                            for pragma in intf[module][zone]:
                                if key == 'top_module':
                                    target_module_label = module
                                else:
                                    target_module_label = key + "::" + module
                                temp = self.combinational_pragma_gen(target_module_label, pragma)

                                if parse_zone and pragma['type'] == 'fix':
                                    fixed_pragmas.append(temp[1])
                                elif parse_zone and pragma['type'] == 'var':
                                    variable_pragmas.append(temp)

        total_comb = self.find_total_pragma_combinations(variable_pragmas)
        print("PYTHON : Total possible directive combinations  = {}".format(total_comb))
        return fixed_pragmas, variable_pragmas, total_comb

    def create_fixed_directive_tcl_file(self, directive_type, sol=''):

        ctrl_intf = {'hs': "set_directive_interface -mode ap_ctrl_hs \"{}\"".format(self.cfg.design_setting.topmodule),
        'none': "set_directive_interface -mode ap_ctrl_none \"{}\"".format(self.cfg.design_setting.topmodule),
        'axi':"set_directive_interface -mode s_axilite \"{}\"".format(self.cfg.design_setting.topmodule)}

        tcl_lines = []
        tcl_lines.append("############################################################")
        tcl_lines.append(
            "## This file is generated automatically by dnn tool. This is {} Solution.".format(directive_type))
        tcl_lines.append("############################################################")

        tcl_lines.append(ctrl_intf.get(self.cfg.design.module_controller,''))

        for intf in self.cfg.interface[self.cfg.design.data_interface]:
            tcl_lines.append(intf)

        if not self.cfg.pragmas[directive_type] == None:
            for item in self.cfg.pragmas[directive_type]:
                if item is not None:
                    tcl_lines.append(item)

        if self.cfg.design.dataflow:
            tcl_lines.append('\nset_directive_dataflow    \"{}\"'.format(self.cfg.design_setting.topmodule))
        if sol == '':
            filename = os.path.join(self.cfg.paths.solution, "directives.tcl")
        else:
            filename = os.path.join(sol, "directives.tcl")

        self.utils.save_list_to_file(filename, tcl_lines)
        return tcl_lines

    def read_impl_results(self, sol_path):
        power = {}
        try:
            power_indx = {'Total': 3, 'Static':4, 'DSPs': 3, 'Block': 4,  'Slice': 4,  'Signals': 3,  'Clocks': 3}
            power_rptFile = os.path.join(sol_path, 'impl', 'verilog', 'report', 'rpt_power.xml')
            #power_rpt = pd.read_fwf(power_rptFile, skiprows=list(range(31)), nrows=45 - 31 - 1)
            power_rpt = pd.read_fwf(power_rptFile, skiprows=list(range(53)), nrows=69 - 54 - 1)
            power_rpt2 = power_rpt.T.values.flatten()
            for line, words in enumerate(power_rpt2):
                wordsS = words.split()
                target = wordsS[1]
                if target in power_indx.keys():
                    power['P_{}'.format(target)] = float(wordsS[power_indx[target]]) * 1000
                    #print('Power {}:{}'.format(target, wordsS[power_indx[target]]))
                if target == 'Total': break
        except:
            power = {'P_Clocks': 'NF', 'P_Signals': 'NF', 'P_Slice': 'NF', 'P_Block': 'NF', 'P_DSPs': 'NF', 'P_Static': 'NF', 'P_Total': 'NF'}

        try:
            utilization_rptFile = os.path.join(sol_path, 'impl', 'report', 'verilog',
                                               '{}_export.xml'.format(self.cfg.design_setting.topmodule))
            tree = ET.parse(utilization_rptFile)
            root = tree.getroot()
            PR_results = {i.tag+'_PS': i.text for i in root[0][0]}
            #PR_results['power'] = power['P_total']
            PR_results['Timing_PS'] = root[1][1].text
            PR_results.pop('SLICE_PS')
            PR_results.pop('SRL_PS')
            print('Post Syn hardware utilization : ', PR_results)
        except:
            PR_results = {'LUT_PS': 'NF', 'FF_PS': 'NF', 'DSP_PS': 'NF', 'BRAM_PS': 'NF', 'Timing_PS':'NF'}
        return PR_results, power

    def read_parallel_syn_results(self, solution_num, syn_exec_time, print_out=False):
        file = os.path.join(self.cfg.paths.design_model,'hls{}'.format(solution_num),self.cfg.design_setting.solution_name,
                                '{}_data.json'.format(self.cfg.design_setting.solution_name))
        try:
            with open(file) as json_file:
                json_data = json.load(json_file)
                if self.cfg.design_setting.vivado_version == 2020:
                    passed_sol = self.extract_hls_json_info_vitis(json_data)
                else:
                    passed_sol = self.extract_hls_json_info(json_data)
                model_layers_name = passed_sol.keys()
                passed_sol['solution'] = solution_num
                passed_sol['syn_status'] = 'passed'
                passed_sol['syn_time'] = '{:3d}:{:2d}'.format(syn_exec_time[0], syn_exec_time[1])
                json_file.close()
                if print_out:
                    pprint.pprint(passed_sol[self.cfg.design_setting.topmodule], depth=5)

        except IOError:
            passed_sol = {}
            model_layers_name = []
            passed_sol['solution'] = solution_num
            passed_sol['syn_status'] = 'failed'
            passed_sol['syn_time'] = '{:3d}:{:2d}'.format(syn_exec_time[0], syn_exec_time[1])
            print("PYTHON : can't open {} file. Synthesize failed".format(file))
            model_layers_name = ''
        return passed_sol, model_layers_name

    def read_single_syn_results(self, solution_num, syn_exec_time, print_out=False):
        try:
            with open(self.cfg.files.SolutionFile) as json_file:
                json_data = json.load(json_file)
                json_file.close()
                if self.cfg.design_setting.vivado_version == 2020:
                    passed_sol = self.extract_hls_json_info_vitis(json_data)
                else:
                    passed_sol = self.extract_hls_json_info(json_data)
                model_layers_name = list(passed_sol.keys())
                passed_sol['solution'] = solution_num
                passed_sol['syn_status'] = 'passed'
                passed_sol['syn_time'] = '{:3d}:{:2d}'.format(syn_exec_time[0], syn_exec_time[1])

                if print_out:
                    print('The synthesize result is as below: ')
                    print(self.utils.print_dict(passed_sol[self.cfg.design_setting.topmodule], 4, 30, 4, ' '))

        except IOError:
            print("PYTHON : can't open {} file".format(self.cfg.files.SolutionFile))
            passed_sol = {}
            model_layers_name = ''
            passed_sol['solution'] = solution_num
            passed_sol['syn_status'] = 'failed'
            passed_sol['syn_time'] = '{:3d}:{:2d}'.format(syn_exec_time[0], syn_exec_time[1])
        return passed_sol, model_layers_name

    def extract_hls_json_info(self, json_data):
        syn_rslt_summary = {}
        keys = ['Area', 'Latency']
        temp = json_data["ModuleInfo"]["Metrics"]
        topmodule = self.cfg.design_setting.topmodule
        for p_id, p_info in temp.items():
            syn_rslt_summary[p_id] = {}
            # print("\nModule name:", p_id)
            for key in p_info:
                if key == "Area" and key in keys:
                    a = syn_rslt_summary[p_id]['BRAM %'] = str(
                        round(int(temp[p_id][key]["BRAM_18K"]) / self.cfg.FPGA.BRAM * 100, 2))
                    syn_rslt_summary[p_id]['BRAM'] = temp[p_id][key]["BRAM_18K"]
                    b = syn_rslt_summary[p_id]['LUT %'] = str(
                        round(int(temp[p_id][key]["LUT"]) / self.cfg.FPGA.LUT * 100, 2))
                    syn_rslt_summary[p_id]['LUT'] = temp[p_id][key]["LUT"]
                    c = syn_rslt_summary[p_id]['FF %'] = str(round(int(temp[p_id][key]["FF"]) / self.cfg.FPGA.FF * 100, 2))  # ...
                    syn_rslt_summary[p_id]['FF'] = temp[p_id][key]["FF"]
                    syn_rslt_summary[p_id]['DSP'] = temp[p_id][key]["DSP48E"]
                    d = syn_rslt_summary[p_id]['DSP %'] = str(
                        round(int(temp[p_id][key]["DSP48E"]) / self.cfg.FPGA.DSP * 100, 2))
                    syn_rslt_summary[p_id]['Total %'] = str(round(
                        (float(a) + float(b) + float(c) + float(d)) / 4, 2))
                elif key == "Latency" and key in keys:
                    syn_rslt_summary[p_id]['latency'] = round(int(temp[p_id][key]["LatencyWorst"]) / self.cfg.design_setting.dev_etime)
                    exec_us = (float(temp[p_id][key]["LatencyWorst"])) * float(temp[p_id]["Timing"]["Target"]) / pow(10, 3)
                    syn_rslt_summary[p_id]['exec us'] = str(round(exec_us, 2))
                    syn_rslt_summary[p_id]['clock period'] = float(temp[p_id]["Timing"]["Target"])
                    syn_rslt_summary[p_id]['FPS'] = int(pow(10, 6)/exec_us)
                    if topmodule == p_id:
                        total_OP = self.cfg.analyze_results[topmodule].get('ops', 0)
                        syn_rslt_summary[p_id]['ratio'] = round(self.cfg.analyze_results[topmodule]['ops'] /
                                                                int(syn_rslt_summary[p_id]['latency']), 2)
                    else:
                        total_OP = 0
                        syn_rslt_summary[p_id]['ratio'] = 0
                    syn_rslt_summary[p_id]['OP'] = total_OP
                    syn_rslt_summary[p_id]['GOPS'] = round(total_OP / (exec_us * pow(10, 3)), 3)

                elif key == "Loops" and key in keys:
                    for II in temp[p_id][key]:
                        syn_rslt_summary[p_id]['II {}'.format(II["Name"])] = II["PipelineII"]
        return syn_rslt_summary

    def extract_hls_json_info_vitis(self, json_data):
        syn_rslt_summary = {}
        keys = ['Area', 'Latency', 'Loops']
        temp = json_data["ModuleInfo"]["Metrics"]
        topmodule = self.cfg.design_setting.topmodule
        new_temp = {}
        for i in temp.keys():
            if i == self.cfg.design_setting.topmodule:
                new_label = i
            else:
                new_label = '_'.join(i.split('_')[0:2]+[i.split('_')[-3]])
            new_temp[new_label] = temp[i]
        for p_id, p_info in new_temp.items():
            tt = p_id.split('_')
            p_id_T2 = '{}_{}'.format(tt[-1], lyr_syn2map.get('_'.join(tt[0:-1]),''))
            syn_rslt_summary[p_id] = {}
            # print("\nModule name:", p_id)
            for key in p_info:
                if key == "Area" and key in keys:
                    a = syn_rslt_summary[p_id]['BRAM %'] = str(
                        round(int(new_temp[p_id][key]["BRAM_18K"]) / int(new_temp[p_id][key]["AVAIL_BRAM"]) * 100, 2))
                    syn_rslt_summary[p_id]['BRAM'] = new_temp[p_id][key]["BRAM_18K"]
                    b = syn_rslt_summary[p_id]['LUT %'] = str(
                        round(int(new_temp[p_id][key]["LUT"]) / int(new_temp[p_id][key]["AVAIL_LUT"]) * 100, 2))
                    syn_rslt_summary[p_id]['LUT'] = new_temp[p_id][key]["LUT"]
                    c = syn_rslt_summary[p_id]['FF %'] = str(round(int(new_temp[p_id][key]["FF"]) / int(new_temp[p_id][key]["AVAIL_FF"]) * 100, 2))  # ...
                    syn_rslt_summary[p_id]['FF'] = new_temp[p_id][key]["FF"]
                    syn_rslt_summary[p_id]['DSP'] = new_temp[p_id][key]["DSP"]
                    d = syn_rslt_summary[p_id]['DSP %'] = str(
                        round(int(new_temp[p_id][key]["DSP"]) / int(new_temp[p_id][key]["AVAIL_DSP"]) * 100, 2))
                    syn_rslt_summary[p_id]['Total %'] = str(round(
                        (float(a) + float(b) + float(c) + float(d)) / 4, 2))
                elif key == "Latency" and key in keys:
                    syn_rslt_summary[p_id]['latency'] = str(new_temp[p_id][key]["LatencyWorst"])
                    exec_us = (float(new_temp[p_id][key]["LatencyWorst"])) * float(new_temp[p_id]["Timing"]["Target"])/pow(10, 3)
                    syn_rslt_summary[p_id]['exec us'] = str(round(exec_us, 2))
                    syn_rslt_summary[p_id]['clock period'] = float(new_temp[p_id]["Timing"]["Target"])
                    syn_rslt_summary[p_id]['FPS'] = int(pow(10, 6)/exec_us) if exec_us != 0 else 'NA'
                    if p_id_T2 in self.cfg.analyze_results.keys():
                        OPs = self.cfg.analyze_results[p_id_T2].get('ops', 0)
                    elif p_id == self.cfg.design_setting.topmodule:
                        OPs = self.cfg.analyze_results[p_id].get('ops', 0)
                    else:
                        OPs = 0
                    syn_rslt_summary[p_id]['ratio'] = round(OPs / int(syn_rslt_summary[p_id]['latency']), 2)
                    syn_rslt_summary[p_id]['OP'] = OPs
                    syn_rslt_summary[p_id]['GOPS'] = round(OPs / (exec_us * pow(10, 3)), 3) if exec_us != 0 else 'NA'

                elif key == "Loops" and key in keys:
                    for II in new_temp[p_id][key]:
                        syn_rslt_summary[p_id]['II {}'.format(II["Name"])] = II["PipelineII"]
                        syn_rslt_summary[p_id]['Dep {}'.format(II["Name"])] = II["PipelineDepth"]
                        syn_rslt_summary[p_id]['Trip {}'.format(II["Name"])] = II["TripCount"]
                        syn_rslt_summary[p_id]['Latency {}'.format(II["Name"])] = II["Latency"]
        return syn_rslt_summary


    def copy_hls_bc_files(self, sol_counter, specifier = None):
        '''
        Copy all relevant bc files from the current hls run to the reports folder
        :param sol_counter: solution number used for filenames
        :param specifier: (optional) additional specifier used for filenames
        :return: None
        '''
        if self.cfg.design_setting.DSE_setting['copy_bc_files']:
            bc_report_path = os.path.join(self.cfg.paths.dse_report, 'bcfiles')
            if not os.path.exists(bc_report_path):
                os.mkdir(bc_report_path)
            temp = os.path.join(bc_report_path, 'bc_design{}'.format(sol_counter))
            if specifier is not None:
                temp = temp + '_{}'.format(specifier)
            if not os.path.exists(temp):
                os.mkdir(temp)
            bc_path = os.path.join(self.cfg.paths.design_model, 'hls{}'.format(sol_counter),
                                   self.cfg.design_setting.solution_name, '.autopilot', 'db')
            try:
                for file in glob.iglob(os.path.join(bc_path, 'a.o.3.bc')):
                    shutil.copy2(file, temp)
                for file in glob.iglob(os.path.join(bc_path, '*.{}'.format('verbose.bind.rpt'))):
                    shutil.copy2(file, temp)
                for file in glob.iglob(os.path.join(bc_path, '*.{}'.format('verbose.rpt'))):
                    shutil.copy2(file, temp)
                # Create a .zip of the bc files and remove the temp folder
                shutil.make_archive(temp, 'zip', root_dir=temp)
                for file in glob.glob(os.path.join(temp, '*')):
                    os.remove(file)
                os.rmdir(temp)
            except:
                print("WARNING : BC files are not copied !")
        return


    def save_syn_records(self, syn_rslt, postSyn, time):
        record_file = os.path.join(self.cfg.paths.report, '{}'.format('syn_records'))
        try:
            with open(record_file+'.pickle', 'rb') as f:
                previous_records = pickle.load(f)
                print("Record file is updated.")
        except IOError:
            previous_records = []
            print("Record file not exist! A new record file is created.")

        record = {}
        record['syntime'] = '{}:{}'.format(time[0], time[1])
        record['hls_tool'] = self.cfg.design_setting.vivado_version
        record['syn_label'] = self.cfg.design_setting.syn_label
        record['sol'] = self.cfg.design_setting.syn_directive_type
        record['dataflow'] = self.cfg.design.dataflow

        record.update(syn_rslt[0][self.cfg.design_setting.topmodule])
        record.update(postSyn)
        previous_records.append(record)
        with open(record_file+'.pickle', 'wb') as f:
            pickle.dump(previous_records, f)
        keys = previous_records[0].keys()
        with open(record_file+'.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(previous_records)

        txtlines = []
        temp = ''
        for i in previous_records[0].keys():
            temp = temp + '{:^11} '.format(i)
        txtlines.append(temp)
        for rec in previous_records:
            temp = ''
            for i in rec.keys():
                temp = temp + '{:^11} '.format(str(rec[i]))
            txtlines.append(temp)
        self.utils.save_list_to_file(record_file+'.txt', txtlines)


    def prepare_hw_modules(self):
        run = self.cfg.design_setting.HW_setting['execute'].lower()
        if run == 'none':
            return False
        if os.path.exists(self.cfg.paths.integration):
            shutil.rmtree(self.cfg.paths.integration)
        src = os.path.join(self.cfg.paths.solution,'impl','ip')
        dest = os.path.join(self.cfg.paths.integration,'ip')
        shutil.copytree(src,dest)
        dest = os.path.join(self.cfg.paths.integration,'test_files')
        shutil.copytree(self.cfg.paths.test_files,dest)


    def createNrun_hw_architecture(self):
        bd_series = {'ZYNQ':'ZYNQ',
                     'AXI_BRAM': 'AXI_BRAM',
                     'HS':'HS'}
        run = self.cfg.design_setting.HW_setting['execute'].lower()
        bd_type = bd_series.get(self.cfg.design_setting.HW_setting['bd_type'], False)
        if run == 'none':
            return False

        print("PYTHON : Started mapping accelerator on FPGA:")
        power_report = self.cfg.design_setting.HW_setting['power_report']
        extension_name = self.cfg.design_setting.HW_setting['extension_name']
        self.cfg.paths.report_hw = "{}_hw/report_hw".format(self.cfg.design_setting.design_model)
        report_hw_utilization = os.path.join(self.cfg.paths.report_hw,'utilization_report_syn.rpt')
        report_hw_power = os.path.join(self.cfg.paths.report_hw, 'power_report_syn.rpt')
        fpga_chip = self.cfg.FPGA.board_part
        if not (bd_type and fpga_chip):
            print("ERROR: Please correct these cfg: bd_type, fpga_chip")
            print("ERROR: Can't generate DNN HW")
            return
        lines = []
        lines.append("############################################################")
        lines.append("## This file is generated automatically by hls tool")
        lines.append("############################################################")
        lines.append("set toplevel_hls {}".format(self.cfg.design_setting.design_model))
        lines.append("set toplevel_hw {}_hw_{}".format(self.cfg.design_setting.design_model,extension_name))
        lines.append("set input_bram_name  \"bram_input_data\"")
        lines.append("set kernel_bram_name \"bram_kernel\"")
        lines.append("set output_bram_name \"bram_out_data\"")
        lines.append("set sol_name \"{}\"".format(self.cfg.design_setting.solution_name))
        lines.append("set AXI_CLK    {}".format(self.cfg.FPGA.clock_freq))
        lines.append("set RUN_SIM    {}".format(int(run in ['sim','all', 'syn'])))
        lines.append("set RUN_SYN    {}".format(int(run in ['syn', 'impl', 'all'])))
        lines.append("set RUN_IMPL   {}".format(int(run in ['impl', 'all'])))
        lines.append("set RUN_PWR    {}".format(int(power_report)))
        lines.append("set RUN_BIT    {}".format(int(run in ['all'])))
        lines.append("set AXI_MODE \"{}\"".format(bd_type))
        lines.append("set FPGA_chip  {}".format(fpga_chip))

        self.utils.save_list_to_file(os.path.join(self.cfg.paths.integration,'dnn_hw_gen_cfg.tcl'), lines)

        version = self.cfg.design_setting.vivado_version
        vivado_cmd = self.cfg.vivado_path[sys.platform][version]['VIVADO']

        if bd_type == 'HS':
            cmd = '{}  -notrace -mode batch -source src/dnn_hw_gen_shared_mem_ultra_hs.tcl -tclargs {} > hw_gen.log || exit $?'.format(vivado_cmd, self.cfg.design_setting.design_model)
        else:
            cmd = '{}  -notrace -mode batch -source src/dnn_hw_gen_shared_mem_ultra.tcl -tclargs {} > hw_gen.log || exit $?'.format(vivado_cmd, self.cfg.design_setting.design_model)

        sr = os.system(cmd)
        beep('impl')
        beep('impl')
        labels_indx = {"DSPs":3,
                       "Slice LUTs" : 4,
                       "Slice Registers": 4,
                       "Block RAM": 5}
        label_replacer = {"DSPs":'DSP_V',
                       "Slice LUTs" : "LUT_V",
                       "Slice Registers": 'FF_V',
                       "Block RAM": 'BRAM_V'}
        results = {}
        try:
            with open(report_hw_utilization, 'r') as file:
                for line in file:
                    temp = line.split()
                    if temp != []:
                        a = temp[0]
                        for key in labels_indx:
                            if line.find(key) > 0 and a == '|':
                                results[label_replacer[key]] = temp[labels_indx[key]]

            power_indx = {"Static Power": 4,
                           "DSPs": 3,
                           "dnn_top_acc_i": 3,
                           "Total On-Chip": 6}
            power_replacer = {"Static Power": "P_Static",
                           "DSPs": "P_DSP",
                           "dnn_top_acc_i": "P_DNN",
                           "Total On-Chip": "P_Total"}
            with open(report_hw_power, 'r') as file:
                for line in file:
                    temp = line.split()
                    if temp != []:
                        a = temp[0]
                        for key in power_indx:
                            if line.find(key) > 0 and a == '|':
                                results[power_replacer[key]] = temp[power_indx[key]]
            beep('impl')
        except:
            results = {}
        return results


#print("PYTHON : hls_tools is imported")