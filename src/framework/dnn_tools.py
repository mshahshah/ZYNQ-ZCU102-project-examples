# ////////////////////////////////////////////////////////////////////////////////////////
# Copyright (c) by
# Company:  IDEA LAB, The University of Texas at Dallas
# Author :  Masoud Shahshahani
#
# Originally Create Date: Mar-5, 2020
# Project Name    : DNN_Framework
# Tool Versions   : Python +3
#
# Description     : Configure and Build C/C++ codes based on the given network size
# Dependencies    :
# Additional Comments:
#
# ///////////////////////////////////////////////////////////////////////////////////////

import os, shutil
import csv, time
from shutil import copyfile
import sys, glob, json, random, pickle
import numpy as np
import yaml , math
import subprocess, copy
cur_path = os.getcwd()
sys.path.append('/home/eng/m/mxs161831/Desktop/dnn_small')
import pickle

from utils import utils
from utils import Struct
from dnn_analyzer import *
from dnn_DSE import *



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

class configure_design:
    '''
    configure_design contains functions used to read input yaml files, configure options, and manage the file system
    '''
    def __init__(self):
        self.hello=0

    def create_cfg(self,options):
        '''
        Parse the input_arguments.yaml and design_name.yaml files and store all relevant information in
        a dictionary object cfg
        :param options: Python parameters, passed by the user at runtime
        :return: cfg
        '''
        cfg={}
        self.options = options
        t1 = self.parse_yaml_input_arguments()
        cfg.update(t1)
        self.cfg = Struct(**cfg)

        cfg['run_options'] = options
        [cfg['paths'], cfg['files']] = self.create_paths()
        self.cfg = Struct(**cfg)

        t2 = self.parse_yaml_design_arguments()
        cfg.update(t2)
        self.cfg = Struct(**cfg)

        t3 = {}
        t3['analyze_results'] = {}
        cfg.update(t3)
        self.cfg = Struct(**cfg)

        return self.cfg

    def parse_yaml_design_arguments(self):
        datamap_dict = {}
        with open(os.path.join('cnn_models', '{}.yaml'.format(self.cfg.design_setting.design_model))) as f:
            datamap = yaml.safe_load(f)
            datamap = Struct(**datamap)
            datamap_dict['design'] = Struct(**datamap.design)
            datamap_dict['design_layers'] = datamap.design_layers
            datamap_dict['design_variable_types'] = datamap.design_variable_types
            datamap_dict['pragmas'] = {}
            datamap_dict['pragmas']['variable'] = datamap.pragmas
            datamap_dict['pragmas']['custom'] = datamap.custom_pragma_list
            datamap_dict['pragmas']['best'] = datamap.best_pragma_list
            datamap_dict['pragmas']['base'] = datamap.base_pragma_list
            datamap_dict['pragmas']['minimal'] = datamap.minimal_pragma_list
            datamap_dict['pragmas']['none'] = datamap.none
            datamap_dict['interface'] = datamap.interface
        return datamap_dict

    def parse_yaml_input_arguments(self):
        datamap_dict={}
        fpga_chips = self.parse_yaml_FPGA_chips()
        temp = {}
        with open('input_arguments.yaml') as f:
            datamap = yaml.safe_load(f)
            datamap['FPGA']['clock_freq'] = round(1000 / datamap['FPGA']['clock_period'])
            datamap['design_setting']['topmodule'] = 'dnn_{}'.format(datamap['design_setting']['design_model'])
            datamap = Struct(**datamap)

            datamap_dict['design_setting'] = Struct(**datamap.design_setting)
            #
        if datamap.FPGA['chip'] in fpga_chips.keys():
            for i in ['DSP', 'BRAM', 'LUT', 'FF', 'part', 'board_part']:
                datamap.FPGA[i] = fpga_chips[datamap.FPGA['chip']][i]
            datamap_dict['FPGA'] = Struct(**datamap.FPGA)
            datamap_dict['vivado_path'] = datamap.vivado_path
        else:
            IOError('Select a right FPGA chip in input_arguments!')

        return datamap_dict

    def parse_yaml_FPGA_chips(self):
        datamap_dict={}
        with open('src/fpga_resources.yaml') as f:
            datamap = yaml.safe_load(f)
            #for i in datamap.keys():
            #    datamap_dict[i] = Struct(**datamap[i])
        return datamap

    def create_paths(self):
        '''
        Generates a dictionary of relevant directory paths and file paths for the design project
        :return: paths, files
        '''
        paths={}
        files={}
        paths['design_top'] = os.getcwd()
        if self.options.mode in ['evaluate_model','evaluate_model_report']:
            paths['design_model'] = os.path.join(paths['design_top'], 'cnn_models', 'all_models_evaluation')
        else:
            paths['design_model'] = os.path.join(paths['design_top'], 'cnn_models', self.cfg.design_setting.design_model)
        # paths['python'] = os.path.join(paths['design_top'], 'dnn_python')
        paths['hls'] = os.path.join(paths['design_model'], 'hls')
        paths['solution'] = os.path.join(paths['hls'], self.cfg.design_setting.solution_name)
        paths['directive_list'] = os.path.join(paths['design_model'], self.cfg.design_setting.solution_name + '_sol_list')
        paths['hw'] = os.path.join(paths['design_model'], 'dnn_hw')
        paths['src'] = os.path.join(paths['design_top'], 'src')
        paths['test_data'] = os.path.join(paths['design_top'], 'test_data')
        paths['dse'] = os.path.join(paths['design_model'], 'dnn_DSE')
        paths['report'] = os.path.join(paths['design_model'], 'reports')
        paths['sim'] = os.path.join(paths['design_model'], 'sim')
        paths['test_files'] = os.path.join(paths['sim'], 'test_files')
        paths['rtl_out'] = os.path.join(paths['sim'], 'rtl_out')
        paths['hls_out'] = os.path.join(paths['sim'], 'hls_out')
        paths['ml_dataset'] = os.path.join(paths['design_top'], 'training_dataset')
        #paths['dse_report'] = os.path.join(paths['report'], self.cfg.design_setting.DSE_setting['dse_name'])
        if self.options.mode.split('_')[-1] == 'report':
            mode = '_'.join(self.options.mode.split('_')[0:-1])
        else:
            mode = self.options.mode

        if mode in ['syn','all']:
            rpt_path_name = 'syn'
        elif mode in ['dse_pragma', 'dse_universal', 'dse_pragma_clock']:
            rpt_path_name = mode + '_' + str(self.cfg.design_setting.DSE_setting['solution_counts']) +\
                            '_' + self.cfg.design_setting.DSE_setting['directive_selection'] +\
                            '_' + self.cfg.design_setting.DSE_setting['dse_name']

        elif mode in ['dse_clk_pragma_cfg']:
            rpt_path_name = mode + '_' + str(self.cfg.design_setting.DSE_setting['solution_counts'] *
                                             self.cfg.design_setting.DSE_setting['config_count']) + \
                                   '_' + self.cfg.design_setting.DSE_setting['directive_selection'] +\
                                   '_' + self.cfg.design_setting.DSE_setting['dse_name']

        elif mode in ['dse_clock', 'dse_dtype', 'dse_cfg']:
            rpt_path_name = mode + '_' + self.cfg.design_setting.syn_directive_type +\
                '_' + self.cfg.design_setting.DSE_setting['dse_name']
        elif mode in ['evaluate_model']:
            rpt_path_name = self.cfg.design_setting.syn_directive_type + \
                            '_' + self.cfg.design_setting.DSE_setting['dse_name']
        else:
            rpt_path_name = 'unknown_report'
        paths['integration'] = os.path.join(paths['design_model'], 'integration')
        paths['dse_report'] = os.path.join(paths['report'], rpt_path_name)
        paths['dse_figures'] = os.path.join(paths['dse_report'],  'figures')
        paths['dse_pickles'] = os.path.join(paths['dse_report'],  'pickles')
        paths['dse_analyzes'] = os.path.join(paths['dse_report'], 'analyzes')
        paths['dse_json'] = os.path.join(paths['dse_report'],  'json')
        paths['dnn_weights'] = os.path.join(paths['sim'], 'dnn_weights')
        files['synLogFile'] = os.path.join(paths['solution'] , '{}.log'.format(self.cfg.design_setting.solution_name))
        files['SolutionFile'] = os.path.join(paths['solution'], '{}_data.json'.format(self.cfg.design_setting.solution_name))
        files['DirectiveFile'] = os.path.join(paths['solution'], 'directives.tcl')
        files['TopModuleRptFile'] = os.path.join(paths['solution'],'syn','report','{}_csynth.rpt'.format(self.cfg.design_setting.topmodule))
        files['user_defined_arguments'] = os.path.join( 'input_arguments.yaml')
        files['user_defined_layers'] = os.path.join(paths['src'],'user_defined_layers.yaml')
        files['dnn_cfg_cppfile'] = os.path.join(paths['design_model'], 'dnn_configs.h')
        files['dnn_main_cppfile'] = os.path.join(paths['design_model'], 'top.cpp')
        files['dnn_main_hfile'] = os.path.join(paths['design_model'], 'top.h')
        files['hw_gen_cfg_file'] = os.path.join(paths['src'], 'dnn_hw_gen_cfg.tcl')
        files['hw_gen_shared_mem'] = os.path.join(paths['src'], 'dnn_hw_gen_shared_mem.tcl')
        files['trained_model_weights_float'] = os.path.join(paths['dnn_weights'], 'model_weights_float')
        files['trained_model_weights_fixed'] = os.path.join(paths['dnn_weights'], 'model_weights_fixed')
        self.paths = Struct(**paths)
        self.files = Struct(**files)
        return Struct(**paths) , Struct(**files)

    def copy_design_source_files(self,extensions):
        '''
        Creates a copy of C++ source template files into the design directory
        :param extensions: List of file extensions to be copied, Usually ['h', 'cpp']
        :return: none
        '''
        for extension in extensions:
            files = glob.iglob(os.path.join(self.paths.src, 'hardware', '*.{}'.format(extension)))
            for file in files:
                if os.path.isfile(file):
                    shutil.copy(file, self.paths.design_model)

    def prepare_design(self, clean=False):
        '''
        Prepare the project folder directories by removing old directories and adding new ones, as needed.
        :return: None
        '''
        if self.options.mode in ['','syn_report', 'dse_pragma_report','dse_dtype_report','dse_clock_report', 'dse_pragma_clock_report',
                                 'dse_clk_pragma_cfg_report','dse_universal_report','dse_variable_report', 'dse_cfg_report','evaluate_model_report']:
            return
        elif self.options.mode in ['evaluate_model', 'syn', 'all'] or 'dse' in self.options.mode:
            if self.cfg.design_setting.run_vivado_power_analyzer: #force vivado synthesizer if power is asked
                self.cfg.design_setting.run_vivado_synthesize = True


            # Create clean subdirectories for the design
            utils.create_directory(self.cfg.paths.solution, True)
            utils.create_directory(self.cfg.paths.test_files, True)
            utils.create_directory(self.cfg.paths.report, False)

            self.copy_design_source_files(['h', 'cpp'])

            if self.options.mode not in ['syn', 'all']:
                utils.create_directory(self.cfg.paths.directive_list, True)
                utils.create_directory(self.cfg.paths.dse_report, True)
                utils.create_directory(self.cfg.paths.dse_figures, True)
                utils.create_directory(self.cfg.paths.dse_pickles, True)
                utils.create_directory(self.cfg.paths.dse_analyzes, True)
                utils.create_directory(self.cfg.paths.dse_json, True)

class dnn_tools:
    def __init__(self,cfg):
        self.cfg = cfg
        self.utils = utils(cfg)
        self.conv_cfg_dict = {'w_in': 0, 'stride': 0, 'w_out': 0, 'rc_strt': 0, 'rc_end': 0, 'w_ker': 0, 'w2_ker': 0,
                              'lyr_in': 0, 'lyr_out': 0, 'mux_mult': 1, 'batch_in': 1, 'batch_out': 1}
        self.fc_cfg_dict = {'lyr_in': 0, 'lyr_out': 0, 'col_max': 0, 'row_max': 0,
                            'fc_ker': 0, 'mux_mult': 1, 'batch_in': 1, 'batch_out': 1}
        self.port_cfg_dict = {'in_mem': 0, 'ker_mem': 0, 'out_mem': 0}
        self.hls_dnn_layers = {'conv': 'conv_3DT1', 'max_pool': 'ds_3DT1', 'avg_pool': 'ds_3DT2', 'fc': 'fc_T1', 'conv2fc':'conv2fc',
                               'read_kernel2d': 'read_kernel2D','read_kernel3d': 'read_kernel3D', 'read_kernel': 'read_kernel',
                               'write_result': 'write_result', 'write_result3D': 'write_result3D', 'read_input3D': 'read_input3D',
                               'read_input': 'read_input'}

        self.lyr_template = {'cfg':'', 'type': '', 'id': 0, 'w_in': 0, 'stride': 1, 'w_out': 0, 'rc_strt': 0, 'rc_end': 0,
                             'w_ker': 0, 'w2_ker': 0,
                             'lyr_in': 0, 'lyr_out': 0, 'mux_mult': 1, 'in_t': '', 'ker_t': '', 'res_t': '',
                             'activation': '', 'pooling':'', 'batch_in': 1, 'batch_out': 1}

    def load_fpga_info(self,yamlfile):
        with open(yamlfile) as f:
            datamap = yaml.safe_load(f)
        fpga_chips = {}
        for a in datamap.keys():
            fpga_chips[a] = Struct(**datamap[a])
            fpga_chips[a].Resources = Struct(**fpga_chips[a].Resources)
        return fpga_chips


    def create_coe_file(self,filename,data,radix ='hex',reshape=1):
        try:
            dataR = np.array(data).reshape(-1, reshape)
        except:
            print('PYTHON : cannot reshape the memory initialization data, size is not correct')

        coe_list=[]
        if   radix == 'dec' :
            header = 'memory_initialization_radix=10;\nmemory_initialization_vector='
            dataR = dataR.astype('str')
            if reshape != 1 :
                print("ERROR PYTHON : cannot reshape the memory initialization data, decimal reshape is not supported")
            for i in dataR: coe_list.append(''.join(i))
        elif radix == 'bin':
            header = 'memory_initialization_radix=2; \nmemory_initialization_vector='
            for i in dataR: coe_list.append(''.join(i))
        else :
            header = 'memory_initialization_radix=16;\nmemory_initialization_vector='
            for i in dataR: coe_list.append(''.join(i))

        coe_list.insert(0,header)
        coe_list.append(';')
        np.savetxt(filename + '.coe', coe_list, fmt='%4s')

    def pars_user_layer_defined(self):
        var_types = self.cfg.design_variable_types
        dnn_cfg_dict = self.cfg.design_layers
        dnn_lyr_list = []
        for lyrNum,lyr in enumerate(dnn_cfg_dict):
            if lyr['type'].upper() in ['IN','CONV','POOL','FC']:
                dnn_lyr_list.append(lyr)
            else:
                print('* PYTHON : dnn_tools : Layer {} type is not supported.'.format(lyrNum))
        tmp = []
        first_FC = True

        for lyrNum in range(len(dnn_lyr_list)):

            lyr_hw = self.lyr_template
            tmp.append(copy.deepcopy(lyr_hw))

        for lyrNum, dnn_lyr in enumerate(dnn_lyr_list):
            if dnn_lyr['type'].upper() == 'IN':
                tmp[lyrNum]['type'] = 'IN'
                tmp[lyrNum]['id'] = lyrNum
                tmp[lyrNum]['in_t'] = dnn_lyr.get('data_type',16)
                tmp[lyrNum]['w_out'] = dnn_lyr['w_out']
                tmp[lyrNum]['lyr_out'] = dnn_lyr['lyr_out']

            elif dnn_lyr['type'].upper() == 'CONV':
                tmp[lyrNum]['type'] = 'CONV'
                tmp[lyrNum]['id'] = lyrNum
                tmp[lyrNum]['in_t'] = dnn_lyr.get('data_type',16)
                tmp[lyrNum]['stride'] = dnn_lyr['stride']
                tmp[lyrNum]['w_ker'] = dnn_lyr['w_ker']
                tmp[lyrNum]['w2_ker'] = tmp[lyrNum]['w_ker']**2
                tmp[lyrNum]['w_in'] = tmp[lyrNum-1]['w_out']
                tmp[lyrNum]['w_out'] = round((tmp[lyrNum]['w_in']-tmp[lyrNum]['w_ker'] + 1)/tmp[lyrNum]['stride'])
                tmp[lyrNum]['lyr_in'] = round(tmp[lyrNum-1]['lyr_out'] / dnn_lyr.get('batch_in',1))
                tmp[lyrNum]['batch_in'] = dnn_lyr.get('batch_in',1)
                tmp[lyrNum]['lyr_out'] = round(dnn_lyr['lyr_out'] / dnn_lyr.get('batch_out',1))
                tmp[lyrNum]['batch_out'] = dnn_lyr.get('batch_out',1)
                tmp[lyrNum]['activation'] = dnn_lyr.get('activation','relu')
                tmp[lyrNum]['rc_strt'] = 0
                tmp[lyrNum]['rc_end'] = tmp[lyrNum]['w_in']-1
                tmp[lyrNum]['mux_mult'] = tmp[lyrNum]['w2_ker'] * tmp[lyrNum]['lyr_out']

            elif dnn_lyr['type'].upper() == 'POOL':
                tmp[lyrNum]['type'] = 'POOL'
                tmp[lyrNum]['id'] = lyrNum
                tmp[lyrNum]['in_t'] = dnn_lyr.get('data_type',16)
                tmp[lyrNum]['w_ker'] = dnn_lyr['w_ker']
                tmp[lyrNum]['w2_ker'] = tmp[lyrNum]['w_ker'] ** 2
                tmp[lyrNum]['w_in']  = tmp[lyrNum-1]['w_out']
                tmp[lyrNum]['stride'] = dnn_lyr['stride']
                tmp[lyrNum]['w_out'] = round(tmp[lyrNum]['w_in']/tmp[lyrNum]['stride'])
                tmp[lyrNum]['pooling'] = dnn_lyr.get('pooling','max')
                tmp[lyrNum]['lyr_in'] = tmp[lyrNum-1]['lyr_out']
                tmp[lyrNum]['lyr_out'] = tmp[lyrNum]['lyr_in']
                tmp[lyrNum]['rc_strt'] = 0
                tmp[lyrNum]['rc_end'] = tmp[lyrNum]['w_in']-tmp[lyrNum]['stride']
                tmp[lyrNum]['mux_mult'] = 0

            elif dnn_lyr['type'].upper() == 'FC':
                tmp[lyrNum]['type'] = 'FC'
                tmp[lyrNum]['id'] = lyrNum
                tmp[lyrNum]['in_t'] = dnn_lyr.get('data_type',16)

                if first_FC and lyrNum == 1:
                    first_FC = False
                    tmp[lyrNum]['lyr_in'] = tmp[lyrNum - 1]['lyr_out']
                elif first_FC:
                    first_FC = False
                    tmp[lyrNum]['lyr_in'] = (tmp[lyrNum - 1]['w_out'] ** 2) * tmp[lyrNum - 1]['lyr_out']
                else:
                    tmp[lyrNum]['lyr_in'] = tmp[lyrNum-1]['lyr_out']
                tmp[lyrNum]['batch_in'] = dnn_lyr.get('batch_in', 1)
                tmp[lyrNum]['col_max'] = round(tmp[lyrNum]['lyr_in'] / tmp[lyrNum]['batch_in'])

                tmp[lyrNum]['lyr_out'] = dnn_lyr['lyr_out']
                tmp[lyrNum]['batch_out'] = dnn_lyr.get('batch_out', 1)
                tmp[lyrNum]['row_max'] = round(tmp[lyrNum]['lyr_out'] / tmp[lyrNum]['batch_out'])

                tmp[lyrNum]['fc_ker'] = round(tmp[lyrNum]['row_max'] * tmp[lyrNum]['col_max'])
                tmp[lyrNum]['activation'] = dnn_lyr.get('activation','max')
        return tmp, var_types

    def print_user_layer_defined(self,given_lyrs):
        print('Below is the given layer details')
        for lyrNum, lyr in enumerate(given_lyrs):
            if lyr['type'].upper() == 'CONV':
                print("\tLayer {}: Convolution -->".format(lyrNum), end = '')
                print("\tIn={}\tOut={}\tKer={}\tStride={}\tlyr_in={}\tlyr_out={}".format(
                    lyr['w_in'],lyr['w_out'], lyr['w_ker'], lyr['stride'], lyr['lyr_in'],lyr['lyr_out']))
            elif lyr['type'].upper() == 'POOL':
                print("\tLayer {}: Pooling     -->".format(lyrNum), end = '')
                print("\tIn={}\tOut={}\tKer={}\tStride={}\tlyr_in={}\tlyr_out={}".format(
                    lyr['w_in'],lyr['w_out'], lyr['w_ker'],lyr['stride'], lyr['lyr_in'], lyr['lyr_out']))

            elif lyr['type'].upper() == 'FC':
                print("\tLayer {}: FC          -->".format(lyrNum), end = '')
                print("\tIn={}\tOut={}".format(
                    lyr['lyr_in'],lyr['lyr_out']))
        print('')

# ================================================================
# ================================================================
    def dnn_config_layers(self, given_lyrs):
        cpp_layers_struct = {}
        lyr_cfg_name_list = []
        for lyrNum, lyr in enumerate(given_lyrs):
            cpp_struct = []
            lyr_cfg_name = 'L{}_'.format(lyrNum) + lyr['type'].upper()
            lyr_cfg_name_list.append(lyr_cfg_name)
            cpp_struct.append("struct {} ".format(lyr_cfg_name) + ' {')
            if lyr['type'].lower() in ['conv', 'pool']:
                for key in self.conv_cfg_dict.keys():
                    a = "\tstatic const unsigned  {:<12} = {:<}; ".format(key, lyr[key])
                    cpp_struct.append(a)
            elif lyr['type'].lower() == 'fc':
                for key in self.fc_cfg_dict.keys():
                    a = "\tstatic const unsigned  {:<12} = {:<}; ".format(key, lyr[key])
                    cpp_struct.append(a)

            cpp_struct.append('\t};\n')
            cpp_layers_struct[lyr_cfg_name] = cpp_struct
        return cpp_layers_struct, lyr_cfg_name_list

    def dnn_configs_ports(self,given_lyrs):
        if given_lyrs[1]['type'].lower() == 'fc':
            mem_in_size  = given_lyrs[0]['w_out']
        else:
            mem_in_size = (given_lyrs[0]['w_out'] ** 2) * given_lyrs[0]['lyr_out']
        if given_lyrs[-1]['type'].lower() == 'fc':
            mem_out_size =  given_lyrs[-1]['lyr_out']
        else:
            mem_out_size = given_lyrs[-1]['lyr_out'] * (given_lyrs[-1]['w_out']**2)

        mem_ker_size = 0
        ker_base_addr = []
        for lyrNum, lyr in enumerate(given_lyrs):
            if lyr['type'].lower() in ['conv']:
                temp = (lyr['w_ker']**2) * lyr['lyr_out']
                mem_ker_size = mem_ker_size + temp
                ker_base_addr.append(mem_ker_size+1)
            elif lyr['type'].lower() in ['fc']:
                if self.cfg.design.fc_shared_memory:
                    mem_ker_size = max(mem_ker_size, int((lyr['lyr_in'] * lyr['lyr_out']) / (lyr['batch_in'] * lyr['batch_out'])))
                else:
                    mem_ker_size = mem_ker_size + int((lyr['lyr_in'] * lyr['lyr_out']) / (lyr['batch_in'] * lyr['batch_out']))
                ker_base_addr.append(mem_ker_size + 1)
            else:
                ker_base_addr.append(0)


        a = "\tstatic const unsigned in_mem   =  {:<}; ".format(mem_in_size)
        b = "\tstatic const unsigned ker_mem  =  {:<}; ".format(mem_ker_size)
        c = "\tstatic const unsigned out_mem  =  {:<}; ".format(mem_out_size)

        cpp_port_struct = []
        cpp_port_struct.append("struct PORT_CFG {")
        cpp_port_struct.append(a)
        cpp_port_struct.append(b)
        cpp_port_struct.append(c)
        cpp_port_struct.append('\t};')
        return cpp_port_struct, ker_base_addr

    def dnn_config_header(self):
        header = []
        header.append("#ifndef __DNN_CONFIGS_H__")
        header.append("#define __DNN_CONFIGS_H__")
        header.append("#include \"top.h\"")
        header.append("#include \"dnn_layers.h\"")
        header.append("using namespace std;")
        return header

    def dnn_config_defines(self):
        defines = []
        defines.append('#ifdef __SYNTHESIS__')
        defines.append('#define NO_PRINT')
        defines.append('#else')
        defines.append('#define MEMORY_PRINT')
        if self.cfg.design_setting.Sim_setting['printTOfile']:
            defines.append('#define ACTIVE_PRINTERS')
        for i in self.cfg.design_setting.Sim_setting['layersTOprint']:
            defines.append('#define {}_PRINT'.format(i.upper()))
        defines.append('#endif')
        defines.append('')
        for i in self.cfg.design_variable_types.keys():
            defines.append('#define  {}_bits  {}'.format(i, self.cfg.design_variable_types[i]))

        return defines

    def combine_dnn_configs_segments(self,given_lyrs,given_var_types):
        dnn_configs = {}
        dnn_configs['header'] = self.dnn_config_header()
        dnn_configs['defines'] = self.dnn_config_defines()
        dnn_configs['data_type'] = data_type_gen('ap_int', given_var_types)
        [dnn_configs['ports'], dnn_configs['base_addr']]  = self.dnn_configs_ports(given_lyrs)
        [dnn_configs['layers'], dnn_configs['cfg']] = self.dnn_config_layers(given_lyrs)
        return dnn_configs

    def create_dnn_configs_file(self,given_lyrs,given_var_types):
        dnn_configs = self.combine_dnn_configs_segments(given_lyrs,given_var_types)
        cpp_lines = []
        cpp_lines.append("// ------------- This is a auto-generated file ----------------------")
        for key in ['header', 'defines', 'data_type', 'ports', 'layers']:
            cpp_lines.append("//----- section : {} -------".format(key))
            if key == 'layers':
                for lyrNum,lyr in enumerate(dnn_configs[key]):
                    cpp_lines.append("//----- {} layer -------".format(lyrNum))
                    for line in dnn_configs[key][lyr]:
                        cpp_lines.append(line)
            elif key == 'data_type':
                temp = ["typedef ap_int < in_t_bits > in_t;",
                 "typedef ap_int < ker_t_bits > ker_t;",
                 "typedef ap_int < res_t_bits > res_t;",
                 "typedef ap_int < mid_t_bits > mid_t;",
                 "typedef ap_int < cfg_t_bits > cfg_t;"]
                #for var in dnn_configs['data_type'].keys():
                #    cpp_lines.append(dnn_configs['data_type'][var])

                for i in temp:
                    cpp_lines.append(i)


            else:
                for line in dnn_configs[key]:
                    cpp_lines.append(line)
            cpp_lines.append("\n")
        cpp_lines.append("\n#endif\n")
        #print("PYTHON : DNN_TOOLS : \"dnn_config.h\" is created!")
        self.utils.save_list_to_file(self.cfg.files.dnn_cfg_cppfile,cpp_lines)

        for lyrNum,lyr in enumerate(given_lyrs):
            lyr['cfg'] = dnn_configs['cfg'][lyrNum]
            lyr['base_addr'] = dnn_configs['base_addr'][lyrNum]
        return given_lyrs

# ================================================================
# ================================================================

    #def conv_cpp_line(self,cfg, id, in_sig, ker_sig, out_sig):
    def conv_cpp_line(self, lyr, var_list, ker_list, lyrNum):
        return "dnn.{:<12} < in_t, ker_t, res_t, mid_t, {:<7} > ( {}, {}, {}, {} );".format(
            self.hls_dnn_layers['conv'], lyr['cfg'], lyr['id'], var_list[lyrNum - 1], ker_list[lyrNum], var_list[lyrNum])

    def ds_cpp_line(self, lyr, var_list, lyrNum):
        if lyr['pooling'] == 'max':
            return "dnn.{:<12} < in_t, res_t, {:<7} >               ( {}, {}, {} );".format(
                self.hls_dnn_layers['max_pool'], lyr['cfg'], lyr['id'], var_list[lyrNum - 1], var_list[lyrNum])
        else:
            return "dnn.{:<12} < in_t, res_t, {:<7} >               ( {}, {}, {} );".format(
                self.hls_dnn_layers['avg_pool'], lyr['cfg'], lyr['id'], var_list[lyrNum - 1], var_list[lyrNum])

    def fc_cpp_line(self, first_fc, lyr, ker_list, var_list, lyrNum):
        if self.cfg.design.fc_shared_memory and not self.cfg.design.dataflow:
            ker_sig = 'kernel_port'
        else:
            ker_sig = ker_list[lyrNum]

        if first_fc:
            in_data = '&{}[0][0][0]'.format(var_list[lyrNum-1]) #var_list[lyrNum-1] + '1D'
        else:
            in_data = var_list[lyrNum-1]
        return "dnn.{:<12} < in_t, ker_t, res_t, mid_t, {:<7} > ( {}, {}, {}, {} );".format(
            self.hls_dnn_layers['fc'], lyr['cfg'], lyr['id'], in_data, ker_sig, var_list[lyrNum])

    def conv2fv_line(self,cfg, id, in_sig, out_sig):
        return "dnn.{:<12} <in_t, {:<7}>                      ( {}, {}, {} );".format(
            self.hls_dnn_layers['conv2fc'], cfg, id, in_sig,  out_sig)

    def read_kernel2D(self, port_cfg, cfg, base_addr, port_sig, out_sig):
        if self.cfg.design.dataflow:
            return ''
        else:
            return "intf.{:<15} <ker_t, {}, {:<7}> ( {:<6}, {}, {} );".format(self.hls_dnn_layers['read_kernel2d'],
                                                                              port_cfg, cfg, base_addr, port_sig,
                                                                              out_sig)

    def read_kernel3D(self, port_cfg, cfg, base_addr, port_sig, out_sig):
        if self.cfg.design.dataflow:
            return ''
        else:
            return "intf.{:<15} <ker_t, {}, {:<7}> ( {:<6}, {}, {} );".format(self.hls_dnn_layers['read_kernel3d'],
                                                                              port_cfg, cfg, base_addr, port_sig,
                                                                              out_sig)

    def read_kernel(self, port_cfg, cfg, base_addr, port_sig, out_sig):
        if self.cfg.design.dataflow:
            return ''
        else:
            return "intf.{:<15} <ker_t, {}, {:<7}> ( {:<6}, {}, {} );".format(self.hls_dnn_layers['read_kernel'],
                                                                              port_cfg, cfg, base_addr, port_sig,
                                                                              out_sig)

    def write_result(self, port_cfg, cfg, port_sig, data_sig):
        return "intf.{:<15} <res_t, {}, {:<7}> ( {}, {} );".format(
            self.hls_dnn_layers['write_result'], port_cfg, cfg, port_sig, data_sig)

    def write_result3D(self, port_cfg, cfg, port_sig, data_sig):
        return "intf.{:<15} <res_t, {}, {:<7}> ( {}, {} );".format(
            self.hls_dnn_layers['write_result3D'], port_cfg, cfg, port_sig, data_sig)

    def read_input(self, port_cfg, cfg, port_sig, data_sig):
        if cfg[1]['type'].lower() == 'fc':
            return "intf.{:<15} <in_t , {}, {:<7}> ( {}, {} );".format(self.hls_dnn_layers['read_input'], port_cfg,
                                                                       cfg[1]['cfg'], port_sig, data_sig)
        else:
            return "intf.{:<15} <in_t , {}, {:<7}> ( {}, {} );".format(self.hls_dnn_layers['read_input3D'], port_cfg,
                                                                       cfg[1]['cfg'], port_sig, data_sig)

    def main_function(self, top_module_name, port_name, dataflow_ker_port):
        if self.cfg.design.dataflow:
            ker_port = ', '.join(dataflow_ker_port)
            temp = "void {}(in_t data_port[{}::in_mem], res_t output_port[{}::out_mem], \n {})".format(
                top_module_name, port_name, port_name, ker_port) + '  \n{ '
            return temp
        else:
            ker_port = 'ker_t kernel_port[{}::ker_mem]'.format(port_name)
            temp = "void {}(in_t data_port[{}::in_mem], {}, res_t output_port[{}::out_mem])".format(
                top_module_name, port_name, ker_port, port_name) + '  \n{ '
            return temp

    def create_main_header_file(self, cpp_segments):
        header_module = cpp_segments['top_module'][0].split(')')[0] + ');'
        ch_lines = ["#ifndef __TOP_H__",
        "#define __TOP_H__",
        "#include <fstream>",
        "#include <iostream>",
        "#include <iomanip>",
        "#include <cstdlib>",
        "#include <cmath>",
        "#include \"ap_fixed.h\"",
        "#include \"ap_int.h\"",
        "#include \"dnn_layers.h\"",
        "#include \"dnn_configs.h\"",
        '', '', '', '', '',
        'void print_configs(void);',
        header_module,
        '', '', '#endif']
        self.utils.save_list_to_file(self.cfg.files.dnn_main_hfile, ch_lines)

    def create_main_cpp_file(self, given_lyrs, predicted_label):
        first_fc = True
        cpp_segments={}
        cpp_segments['headers'] = []
        cpp_segments['top_module'] = []
        cpp_segments['sub_module'] = []
        cpp_segments['variables'] = []
        cpp_segments['debug'] = []
        dataflow_ker_port = []

        variable_list = []
        kernel_list = []
        for lyrNum, lyr in enumerate(given_lyrs):
            variable_list.append('L{}_out'.format(lyrNum))
            kernel_list.append('L{}_ker'.format(lyrNum))
        # ------------------   header lines -------------------------------------------
        cpp_segments['headers'].append('#include \"top.h\"')

        # ------------------   sub-modules  -------------------------------------------
        cpp_segments['sub_module'].append(self.read_input('PORT_CFG', given_lyrs, 'data_port', variable_list[0]))
        if given_lyrs[1]['type'].lower() == 'fc':
            cpp_segments['variables'].append("in_t   {}  [{}::lyr_in];".format(
                variable_list[0], given_lyrs[1]['cfg']))
        else:
            cpp_segments['variables'].append("in_t   {}  [{}::lyr_in][{}::w_in][{}::w_in];".format(
                variable_list[0], given_lyrs[1]['cfg'], given_lyrs[1]['cfg'], given_lyrs[1]['cfg']))
        cpp_segments['variables'].append('')

        # -----------------  create sub_layers for ports -----------------------------
        for lyrNum, lyr in enumerate(given_lyrs):
            if lyr['type'].lower() == 'conv':
                cpp_segments['sub_module'].append(self.read_kernel3D('PORT_CFG', lyr['cfg'], lyr['base_addr'], 'kernel_port', '{}'.format(kernel_list[lyrNum])))
                ker_array = "ker_t  {}  [{}::lyr_out][{}::lyr_in][{}::w2_ker]".format(kernel_list[lyrNum], lyr['cfg'], lyr['cfg'], lyr['cfg'])
                if self.cfg.design.dataflow:
                    cpp_segments['variables'].append('')
                    dataflow_ker_port.append(ker_array)
                else:
                    cpp_segments['variables'].append(ker_array+';')

            elif lyr['type'].lower() == 'fc':
                ker_array = "ker_t  {}  [{}::fc_ker]".format(kernel_list[lyrNum], lyr['cfg'])
                if self.cfg.design.fc_shared_memory or self.cfg.design.dataflow:
                    cpp_segments['sub_module'].append('')
                    cpp_segments['variables'].append('')
                    dataflow_ker_port.append(ker_array)
                else:
                    cpp_segments['sub_module'].append(self.read_kernel('PORT_CFG', lyr['cfg'], lyr['base_addr'], 'kernel_port', '{}'.format(kernel_list[lyrNum])))
                    cpp_segments['variables'].append(ker_array+';')

        # -----------------  create sub dnn layers -----------------------------
        cpp_segments['sub_module'].append('')
        cpp_segments['variables'].append('')
        for lyrNum, lyr in enumerate(given_lyrs):
            if lyr['type'].lower() == 'conv':
                cpp_segments['sub_module'].append(self.conv_cpp_line(lyr, variable_list, kernel_list, lyrNum))
                cpp_segments['variables'].append("res_t  {}  [{}::lyr_out][{}::w_out][{}::w_out];".format(
                    variable_list[lyrNum], lyr['cfg'], lyr['cfg'], lyr['cfg']))

            elif lyr['type'].lower() == 'pool':
                cpp_segments['sub_module'].append(self.ds_cpp_line(lyr, variable_list, lyrNum))
                cpp_segments['variables'].append("res_t  {}  [{}::lyr_out][{}::w_out][{}::w_out];".format(
                    variable_list[lyrNum], lyr['cfg'], lyr['cfg'], lyr['cfg']))
            elif lyr['type'].lower() == 'fc':
                if first_fc and not lyrNum == 1:
                    fc_layer_inst = self.fc_cpp_line(first_fc, lyr, kernel_list, variable_list, lyrNum)
                    #cpp_segments['sub_module'].append(self.conv2fv_line(given_lyrs[lyrNum - 1]['cfg'], lyr['id'], variable_list[lyrNum-1], variable_list[lyrNum-1]+'1D'))
                    cpp_segments['sub_module'].append(fc_layer_inst)
                    temp = "res_t  {}  [{}::lyr_out * {}::w_out * {}::w_out];".format(variable_list[lyrNum - 1] + '1D',
                                                                               given_lyrs[lyrNum - 1]['cfg'],
                                                                               given_lyrs[lyrNum - 1]['cfg'],
                                                                               given_lyrs[lyrNum - 1]['cfg'])
                    #cpp_segments['variables'].append(temp)
                    first_fc = False

                else:
                    first_fc = False
                    fc_layer_inst = self.fc_cpp_line(first_fc, lyr, kernel_list, variable_list, lyrNum)
                    cpp_segments['sub_module'].append(fc_layer_inst)


                cpp_segments['variables'].append("res_t  {}  [{}::lyr_out];".format(variable_list[lyrNum], lyr['cfg']))

        out_lbls = str(predicted_label)

        cpp_segments['debug'].append("res_t  msdtst  [{}::lyr_out]={{{}}};".format(lyr['cfg'], out_lbls[1:-1]))

        # ------------------   Main function  -------------------------------------------
        temp = self.main_function(self.cfg.design_setting.topmodule, 'PORT_CFG', dataflow_ker_port)
        cpp_segments['top_module'].append(temp)
        cpp_segments['top_module'].append("\ninterface  intf;")
        cpp_segments['top_module'].append("dnn_layers dnn;\n")
        # ------------------   write results -------------------------------------------
        cpp_segments['sub_module'].append('')
        if given_lyrs[-1]['type'].lower() == 'fc':
            if self.cfg.design_setting.debug:
                cpp_segments['sub_module'].append('//'+self.write_result('PORT_CFG', given_lyrs[-1]['cfg'], 'output_port', variable_list[-1]))
            else:
                cpp_segments['sub_module'].append(self.write_result('PORT_CFG', given_lyrs[-1]['cfg'], 'output_port', variable_list[-1]))
        elif given_lyrs[-1]['type'].lower() in ['conv', 'pool']:
            cpp_segments['sub_module'].append(
                self.write_result3D('PORT_CFG', given_lyrs[-1]['cfg'], 'output_port', variable_list[-1]))
        cpp_segments['debug'].append(self.write_result('PORT_CFG', given_lyrs[-1]['cfg'], 'output_port', 'msdtst'))
        cpp_lines = []
        cpp_lines.append("// ------------- This is a auto-generated file ----------------------")
        if self.cfg.design_setting.debug:
            segments = ['headers', 'top_module', 'variables', 'sub_module', 'debug']
        else:
            segments = ['headers', 'top_module', 'variables', 'sub_module']
        for segment in segments:
            cpp_lines.append('\n//=====================  {} =====================\n'.format(segment))
            for line in cpp_segments[segment]:
                if segment in ['variables', 'sub_module']:
                    cpp_lines.append('\t'+line)
                else:
                    cpp_lines.append(line)
        cpp_lines.append('\n\n}')
        cpp_lines.append("\n")
        print("PYTHON : DNN_TOOLS : \"top.cpp\" is created!")

        self.utils.save_list_to_file(self.cfg.files.dnn_main_cppfile, cpp_lines)
        return cpp_segments


#print("PYTHON : dnn_tools is imported")

