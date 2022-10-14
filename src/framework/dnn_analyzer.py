# ////////////////////////////////////////////////////////////////////////////////////////
# Copyright (c) by
# Company:  IDEA LAB, The University of Texas at Dallas
# Author :  Masoud Shahshahani
#
# Originally Create Date: Mar-5, 2020
# Project Name    : DNN_Framework
# Tool Versions   : Python +3
#
# Description     : Analyze and report quantitative estimation of the given DNN model
# Dependencies    :
# Additional Comments:
#
# ///////////////////////////////////////////////////////////////////////////////////////

from dnn_tools import *
import os

class dnn_analyzer:
    def __init__(self,cfg):
        self.utils = utils(cfg)
        self.cfg = cfg

    def estimate_conv2D(self, lyr, dataSize):
        pipelineII_dict = {'custom':{3:0, 5:0, 7:0}, 'minimal':{3:5, 5:13, 7:25}, 'best':{3:5, 5:13, 7:25}}
        outSize = lyr['w_out']
        lyr_out = lyr['batch_out'] * lyr['lyr_out']
        each_window_cycles = lyr['w_ker']**2
        pipelineII = pipelineII_dict[self.cfg.design_setting.syn_directive_type][lyr['w_ker']]
        exec_cycles = (outSize **2) * lyr['lyr_out'] * pipelineII #each_window_cycles * lyr_out * lyr['lyr_in']

        kernel_params = (each_window_cycles+1) * lyr['lyr_in'] * lyr_out
        input_params = lyr['w_in']**2 * lyr['lyr_in'] * dataSize
        output_params = outSize**2 * dataSize * lyr_out
        #print("\nLayer #%1d : Conv2D [inSize =%3d][kerSize =%2d][outSize =%3d][stride =%2d][lyr_in =%3d][lyr_out =%3d][reshape =%2d]"
        #      %(id, lyr['w_in'], lyr['w_ker'], outSize, lyr['stride'], lyr['lyr_in'], lyr_out, reshape))
        #print("   Exec cycle = {:,} cycles".format(exec_cycles))
        #print("   BRAM_18K = {:,}".format(bram_18k))
        ops = ((outSize **2)+1) * each_window_cycles * lyr_out * lyr['lyr_in'] * 2
        exp = {}
        exp['ops'] = ops
        exp['latency'] = exec_cycles
        exp['out_bram_18k'] = math.ceil(output_params / (18 * 1024))
        exp['in_bram_18k'] = math.ceil(input_params / (18 * 1024))
        exp['ker_mem_18k'] = math.ceil(kernel_params / (18 * 1024))
        exp['out_mem_bits'] = output_params
        exp['in_mem_bits'] = input_params
        exp['ker_mem_bits'] = kernel_params
        return exp

    def estimate_pooling(self, lyr, dataSize):
        pipelineII = 9
        each_window_cycles = lyr['w_ker']**2
        outSize = round(lyr['w_in']/lyr['w_ker'])
        exec_cycles = round((outSize**2) * pipelineII) # each_window_cycles * lyr['lyr_out'])
        input_params = lyr['lyr_in'] * lyr['w_in'] * lyr['w_in'] * dataSize
        output_params = lyr['lyr_in'] * outSize * outSize * dataSize
        #print("\nlayer #%1d : Pooling [inSize =%3d][outSize =%2d][Layers =%3d][reshape =%2d]" % (id, inSize, outSize, lyr, reshape))
        #print("   Exec cycle = {:,} cycles".format(exec_cycles))
        ops = (outSize **2) * each_window_cycles * lyr['lyr_out']
        exp = {}
        exp['ops'] = ops
        exp['latency'] = exec_cycles
        exp['out_bram_18k'] = math.ceil(output_params / (18 * 1024))
        exp['in_bram_18k'] = math.ceil(input_params / (18 * 1024))
        exp['ker_mem_18k'] = 0
        exp['out_mem_bits'] = output_params
        exp['in_mem_bits'] = input_params
        exp['ker_mem_bits'] = 0
        return exp

    def estimate_fc(self,lyr,  reshape, dataSize):
        ker_size = lyr['lyr_in'] * (lyr['lyr_out']+1) * dataSize
        bram_18k = math.ceil(ker_size / (18 * 1024))
        exp = {}
        exp['ops'] = lyr['lyr_in'] * (lyr['lyr_out']+1) * 2
        exp['latency'] = lyr['lyr_in'] * lyr['lyr_out'] / reshape
        exp['out_bram_18k'] = math.ceil(lyr['lyr_out'] * dataSize / (18 * 1024))
        exp['in_bram_18k'] = math.ceil(lyr['lyr_in'] * dataSize / (18 * 1024))
        exp['ker_mem_18k'] = math.ceil(ker_size / (18 * 1024))
        exp['out_mem_bits'] = lyr['lyr_out'] * dataSize
        exp['in_mem_bits'] = lyr['lyr_in'] * dataSize
        exp['ker_mem_bits'] = ker_size
        #print("\nLayer #%1d : FC [inSize =%3d][lyr['lyr_in'] =%2d][reshape =%2d]" %(id, inSize, lyr['lyr_in'], reshape))
        #print("   Exec cycle = {:,} cycles".format(exec_cycles))
        return exp

    def estimate_input(self, lyr, reshape, dataSize):
        exp = {}
        exp['ops'] = 0
        exp['latency'] = lyr['w_out'] * lyr['w_out'] * lyr['lyr_out'] / reshape
        exp['out_bram_18k'] = math.ceil(lyr['w_out'] * lyr['w_out'] * dataSize * lyr['lyr_out'] / (18 * 1024))
        exp['out_mem_bits'] = lyr['w_out'] * lyr['w_out'] * dataSize * lyr['lyr_out']
        exp['ker_mem_18k'] = 0
        exp['in_mem_bits'] = 0
        exp['ker_mem_bits'] = 0
        #print("\nLayer #%1d : FC [inSize =%3d][lyr['w_out'] =%2d][reshape =%2d]" %(id, inSize, lyr['w_out'], reshape))
        #print("   Exec cycle = {:,} cycles".format(exec_cycles))
        return exp

    def overall_report(self,analyze_results, log=True):
        if log == False : return
        print("PYTHON: Analyzer: The overall report for the Expected Execution Time (EET) of each layer is as below:")
        print("     ------------   Execution time of each layer and top module -------------")
        for lyr in analyze_results.keys():
            print("\t EET of {:<8} =  {:,} clks".format(lyr, analyze_results[lyr]['latency']))
        print("\t Top module has {:6.3f} MOP".format((analyze_results[self.cfg.design_setting.topmodule]['ops']/10**6)))
        print("     ------------   Memory requirements for each layer and top module -------")
        for lyr in analyze_results.keys():
            print("\t intermediate bits of {:<8} =  {:,} ".format(lyr, analyze_results[lyr].get('out_mem_bits',0)))

        #print("     --------------- BRAM 18K requirements for Kernels ----------------------")
        #for lyr in analyze_results.keys():
        #    print("\t Kernel BRAM 18K of {:<8} =  {:,} ".format(lyr, analyze_results[lyr].get('ker_mem_18k',0)))

        #print("     ---------- BRAM 18K requirements for each layer and top module ---------")
        #for lyr in analyze_results.keys():
        #    print("\t intermediate BRAM 18K of {:<8} =  {:,} ".format(lyr, analyze_results[lyr].get('out_bram_18k',0)))
        print('\n')

    def results_deviation(self, syn_results, print_out=True):
        deviation = {}
        analyzes_results = self.cfg.analyze_results[self.cfg.design_setting.topmodule]
        topmodule_syn = syn_results[0][self.cfg.design_setting.topmodule]
        deviation['er_' + 'latency'] = round(abs(analyzes_results['latency'] - int(topmodule_syn['latency'])),0)
        deviation['er_' + 'latency %'] = round((deviation['er_' + 'latency']/analyzes_results['latency']*100), 0)
        if print_out:
            print('\nEST: The synthesize VS analyzes deviation is as below: ')
            print(self.utils.print_dict(deviation, 4, 30, 4, ' '))
        return deviation

    def analyze_given_model(self, given_lyrs):
        self.cfg.analyze_results = {}
        for lyrNum, lyr in enumerate(given_lyrs):
            if lyr['type'].upper() == 'CONV':
                temp = self.estimate_conv2D(lyr, 16)
            elif lyr['type'].upper() == 'POOL':
                temp = self.estimate_pooling(lyr, 16)
            elif lyr['type'].upper() == 'FC':
                temp = self.estimate_fc(lyr, 1, 16)
            elif lyr['type'].upper() == 'IN':
                temp = self.estimate_input(lyr, 1, 16)
            else:
                temp = {'ops': 0, 'latency': 0, 'in_mem_bits': 0, 'out_mem_bits': 0, 'ker_mem_bits': 0}

            self.cfg.analyze_results[lyr['cfg']] = temp
            #self.cfg.analyze_results['out_mem_bits'][lyr['cfg']] = temp.get('out_mem_bits', 0)
        keys = list(self.cfg.analyze_results.keys())
        if self.cfg.design_setting.topmodule in keys : keys.remove(self.cfg.design_setting.topmodule)
        total_ops = sum([self.cfg.analyze_results[lyr]['ops'] for lyr in keys])
        total_exec_cycles = sum([self.cfg.analyze_results[lyr]['latency'] for lyr in keys])
        total_kernel_bits = sum([self.cfg.analyze_results[lyr].get('ker_mem_bits',0) for lyr in keys])
        total_kernel_18Kbram = sum([self.cfg.analyze_results[lyr].get('ker_mem_18k', 0) for lyr in keys])
        total_intermediate_bits = sum([self.cfg.analyze_results[lyr].get('out_mem_bits', 0) for lyr in keys])
        total_intermediate_18Kbram = sum([self.cfg.analyze_results[lyr].get('out_bram_18k', 0) for lyr in keys])
        self.cfg.analyze_results[self.cfg.design_setting.topmodule] = {}
        self.cfg.analyze_results[self.cfg.design_setting.topmodule]['ops'] = total_ops
        self.cfg.analyze_results[self.cfg.design_setting.topmodule]['latency'] = total_exec_cycles
        self.cfg.analyze_results[self.cfg.design_setting.topmodule]['ker_mem_bits'] = total_kernel_bits
        self.cfg.analyze_results[self.cfg.design_setting.topmodule]['ker_mem_18k'] = total_kernel_18Kbram
        self.cfg.analyze_results[self.cfg.design_setting.topmodule]['out_mem_bits'] = total_intermediate_bits
        self.cfg.analyze_results[self.cfg.design_setting.topmodule]['out_bram_18k'] = total_intermediate_18Kbram
        return self.cfg.analyze_results

    def save_overall_report(self, fname='analyzes', plot=False):
        temp = self.cfg.analyze_results
        all_params = {}
        for item in ['ops', 'in_mem_bits', 'ker_mem_bits', 'out_mem_bits']:
            all_params[item] = {}
            for lyr in temp.keys():
                if lyr.split('_')[-1] in ['CONV', 'POOL', 'FC']:
                    all_params[item][lyr] = temp[lyr]['ops']
        path = os.path.join(self.cfg.paths.report, fname)
        self.utils.save_2D_dict_to_csv(path, all_params)
        if plot:
            self.utils.plot_stacked_bar(all_params)



if __name__ == '__main__':
    dnn_analyzer = dnn_analyzer()
    print("PYTHON : Below are the analyzes of the DNN layers")
    exec_cycles = []
    exec_cycles.append(
        dnn_analyzer.estimate_conv2D(inSize=32, kerSize=5, stride=1, lyr_in=1, lyr_out=6, reshape=1))
    exec_cycles.append(dnn_analyzer.estimate_pooling(inSize=28, kerSize=2, lyr=6, reshape=1))
    exec_cycles.append(
        dnn_analyzer.estimate_conv2D(inSize=14, kerSize=5, stride=1, lyr_in=1, lyr_out=16, reshape=1))
    exec_cycles.append(dnn_analyzer.estimate_pooling(inSize=10, kerSize=2, lyr=16, reshape=1))

    exec_cycles.append(dnn_analyzer.estimate_fc(inSize=5 * 5 * 16, outSize=84, reshape=1))
    exec_cycles.append(dnn_analyzer.estimate_fc(inSize=84, outSize=10, reshape=1))
    dnn_analyzer.overall_report(exec_cycles)