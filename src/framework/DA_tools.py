# ////////////////////////////////////////////////////////////////////////////////////////
# Copyright (c) by
# Company:  IDEA LAB, The University of Texas at Dallas
# Author :  Masoud Shahshahani
#
# Originally Create Date: Mar-5, 2020
# Project Name    : DNN_Framework
# Tool Versions   : Python +3
#
# Description     : To be used for analyzing and building any HLS design
# Dependencies    : Vivado 2018 or newer, Python +3
# Additional Comments:
#
# ///////////////////////////////////////////////////////////////////////////////////////

import os, shutil
import csv, time
from shutil import copyfile
import re , argparse
import sys, glob, json, random, pickle
import numpy as np
import yaml, random
import math, re
from xml.dom import minidom
import subprocess
from utils import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from operator import itemgetter
import pprint
from multiprocessing import Process


class configure_design:
    def __init__(self):
        self.hello=0

    def create_cfg(self, options):
        cfg={}
        self.options = options
        t1 = self.parse_yaml_input_arguments()
        cfg.update(t1)
        self.cfg = Struct(**cfg)

        t2 = self.parse_yaml_design_arguments()
        cfg.update(t2)
        self.cfg = Struct(**cfg)

        cfg['run_options'] = options
        [cfg['paths'], cfg['files']] = self.create_paths()
        # [self.paths, self.files] = self.create_paths()
        self.cfg = Struct(**cfg)
        cfg.update(t1)
        self.cfg = Struct(**cfg)
        return self.cfg

    def parse_yaml_design_arguments(self):
        print("parse_yaml_design_arguments")
        datamap_dict = {}
        temp = self.cfg.design_setting.design_model
        print("temp:", temp)
        with open('{}/{}.yaml'.format(temp,temp)) as f:
            print("file: ", f)
            datamap = yaml.safe_load(f)
            datamap = Struct(**datamap)
            datamap_dict['design'] = Struct(**datamap.design)
            datamap_dict['design_variable_types'] = datamap.design_variable_types
            datamap_dict['pragmas'] = {}
            datamap_dict['pragmas']['variable'] = datamap.pragmas
            datamap_dict['pragmas']['custom'] = datamap.custom_pragma_list
            datamap_dict['pragmas']['best'] = datamap.best_pragma_list
            datamap_dict['pragmas']['base'] = datamap.base_pragma_list
            datamap_dict['pragmas']['minimal'] = datamap.minimal_pragma_list
            datamap_dict['pragmas']['none'] = datamap.none
            datamap_dict['interface'] = datamap.interface
            datamap_dict['analyze_results'] = datamap.analyze_results


        return datamap_dict

    def parse_yaml_input_arguments(self):
        datamap_dict={}
        with open('hls_DA.yaml') as f:
            datamap = yaml.safe_load(f)
            datamap = Struct(**datamap)
            datamap_dict['design_setting'] = Struct(**datamap.design_setting)
            datamap_dict['FPGA'] = Struct(**datamap.FPGA)
        return datamap_dict

        cfg = Struct(**datamap)
        self.FPGA = Struct(**cfg.FPGA)
        self.dse_tool = Struct(**cfg.dse_tool)

    def create_paths(self):
        paths={}
        files={}
        paths['design_top'] = os.getcwd()
        paths['design_model'] = os.path.join(paths['design_top'], self.cfg.design_setting.design_model)
        paths['hls'] = os.path.join(paths['design_model'], 'hls')
        paths['solution'] = os.path.join(paths['hls'], self.cfg.design_setting.solution_name)
        paths['directive_list'] = os.path.join(paths['hls'], self.cfg.design_setting.solution_name + '_sol_list')

        paths['hw'] = os.path.join(paths['design_model'], 'hw')
        paths['report'] = os.path.join(paths['design_model'], 'reports')
        paths['dse_report'] = os.path.join(paths['report'], self.cfg.design_setting.DSE_setting['dse_name'])
        paths['dse_figures'] = os.path.join(paths['dse_report'], 'figures')
        files['synLogFile'] = os.path.join(paths['solution'], '{}.log'.format(self.cfg.design_setting.solution_name))
        files['SolutionFile'] = os.path.join(paths['solution'], '{}_data.json'.format(self.cfg.design_setting.solution_name))
        files['DirectiveFile'] = os.path.join(paths['solution'], 'directives.tcl')
        files['TopModuleRptFile'] = os.path.join(paths['solution'],'syn','report','{}_csynth.rpt'.format(self.cfg.design_setting.design_model))
        files['user_defined_arguments'] = os.path.join(paths['design_model'],'{}.yaml'.format(self.cfg.design_setting.design_model))
        self.paths = Struct(**paths)
        self.files = Struct(**files)
        return Struct(**paths) , Struct(**files)

    def prepare_design(self, cleaning = False):
        if self.options.mode in ['','syn_report', 'dse_pragma_report','dse_dtype_report','dse_clock_report','dse_universal_report','dse_variable_report']:
            return
        elif self.options.mode in ['dse_pragma','dse_clock', 'syn','dse_dtype','dse_universal','dse_variable']:

            if os.path.exists(self.cfg.paths.solution):
                shutil.rmtree(self.cfg.paths.solution)
                os.makedirs(self.cfg.paths.solution)
            else:
                os.makedirs(self.cfg.paths.solution)

            if self.options.mode not in ['syn']:
                if not os.path.exists(self.cfg.paths.directive_list):
                    os.makedirs(self.cfg.paths.directive_list)
                else:
                    shutil.rmtree(self.cfg.paths.directive_list)
                    os.makedirs(self.cfg.paths.directive_list)

                if  os.path.exists(self.cfg.paths.dse_report):
                    print("PYTHON : The report folder exist. choose another DSE report folder name!")
                    # exit()
                    shutil.rmtree(self.cfg.paths.dse_report)
                    os.makedirs(self.cfg.paths.dse_report)
                    os.makedirs(self.cfg.paths.dse_figures)
                else:
                    os.makedirs(self.cfg.paths.dse_report)
                    os.makedirs(self.cfg.paths.dse_figures)

print("PYTHON : hls_tools is imported")



class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)