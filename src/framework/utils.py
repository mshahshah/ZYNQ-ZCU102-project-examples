# ////////////////////////////////////////////////////////////////////////////////////////
# Copyright (c) by
# Company:  IDEA LAB, The University of Texas at Dallas
# Author :  Masoud Shahshahani
#
# Originally Create Date: Mar-5, 2020
# Project Name    : DNN_Framework
# Tool Versions   : Python +3
#
# Description     : Utility Functions require for the framework
# Dependencies    :
# Additional Comments:
#
# ///////////////////////////////////////////////////////////////////////////////////////

import numpy as np
import os, shutil
from os import listdir
import csv, time, sys
import pickle,re, argparse
import matplotlib.pyplot as plt
import glob, pandas

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def beep(type):
    if sys.platform == 'linux': 
        return
    import winsound
    if type == 'syn':
        for i in range(3):
            winsound.Beep(1200, 150)
            time.sleep(0.05)
    elif type == 'dse':
        for i in range(2):
            winsound.Beep(1200, 100)
            time.sleep(0.05)
    elif type == 'impl':
        winsound.Beep(2000, 400)
    elif type == 'finish':
        for i in [5,4,3,2]:
            winsound.Beep(1000 + i*100, int(800/i))
            time.sleep(0.01*i)
        winsound.Beep(600, 700)

class utils:
    def __init__(self, cfg):
        self.hello=0
        self.cfg = cfg


    def printDict(self, cfg, num_tabs = 0):
        if(num_tabs == 0):
            print("cfg dict:")
        for k, v in cfg.__dict__.items():
            for t in range(0, num_tabs):
                print("\t", end='')
            print("-", k, ":", v)
            if hasattr(v, "__dict__"):
                utils.printDict(self, v, num_tabs+1)

    def find_Aword_in_file(self,filename,keyword,save_results=True):
        try:
            file = open(filename,'r')
            read=file.readlines()
            file.close()
            count = 0
            detected_list = []
            for lineNum,line in enumerate(read, start = 1):
                if line != '\n' and len(line) > 2: # if line is not blank
                    split = line.split()
                    first_word = line.split()[0].lower()
                    if (keyword.lower() == first_word.strip(":!@()_+=")):
                        count +=1
                        detected_list.append(line)
            if count != 0 :
                print("PYTHON : {}  \"{}\" found in the \"{}\" file".format(count,keyword,filename))
        except FileExistsError:
            print("PYTHON : faild to open {}".format(filename))
        if save_results:
            filename = file = os.path.join(self.cfg.paths.design_top, "{}.log".format(keyword))
            self.save_list_to_file(filename,detected_list)
        return count

    def replace_word_in_file(self, file_path, string_to_replace, replacement_string):
        # Read in the file
        #print('opening file:', file_path)
        with open(file_path, 'r') as file:
            filedata = file.read()

        #print('Replacing', string_to_replace, 'with', replacement_string)
        # Replace the target string
        filedata = filedata.replace(string_to_replace, replacement_string)

        #print('Closing file:', file_path)
        # Write the file out again
        with open(file_path, 'w') as file:
            file.write(filedata)

    def save_list_to_file(self, filename, data):
        with open(filename, 'w') as f:
            for line in data:
                f.write("%s\n" % line)

    def load_file_to_list(self,filename):
        with open(filename, 'r') as reader:
            return reader.readlines()

    def save_dict_to_csv(self, filename, data):
        with open('{}.csv'.format(filename), 'w') as f:
            for key in data.keys():
                f.write("%s,%d\n" % (key, data[key]))

    def save_2D_dict_to_csv(self, filename, data):
        with open('{}.csv'.format(filename), 'w') as f:
            for d1 in data.keys():
                f.write("\n%s\n" % (d1))
                for d2 in data[d1].keys():
                    f.write(" , %s,%d\n" % (d2, data[d1][d2]))

    def record_time(self):
        return time.time()

    def end_and_print_time(self,start_time):
        syn_time = time.time() - start_time
        min, sec = divmod(syn_time, 60)
        #print("PYTHON : Total synthesis time : {:3d} Minutes and {:2d} Seconds".format(int(min), int(sec)))
        return [int(min), int(sec)]

    def end_and_print_time_and_label(self,start_time, label):
        syn_time = time.time() - start_time
        min, sec = divmod(syn_time, 60)
        if min == 0 and sec < 1:
            print("\nTIME : {} in {:3d}M : {:2d}S : {:5.2f}mS ".format(label, int(min), int(sec), float(syn_time*1000)))
        else:
            print("\nTIME : {} in {:3d}M : {:2d}S".format(label, int(min), int(sec)))
        return round(min), round(sec)

    def create_directory(path, clean=False):
        if clean and os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def copy_a_file(self, src, dest):
        shutil.copyfile(src, dest)

    def save_a_variable(self,fname,variable):
        with open(os.path.join(self.cfg.paths.dse_pickles, '{}.pickle'.format(fname)), 'wb') as f:
            pickle.dump(variable, f)

    def load_a_variable(self,variable_name):
        with open(os.path.join(self.cfg.paths.dse_pickles, '{}.pickle'.format(variable_name)), 'rb') as f:
            variable = pickle.load(f)
        return variable

    def list_files_with_ext(self, directory, extension):
        return (f for f in listdir(directory) if f.endswith('.' + extension))

    def save_fig(self,fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(self.cfg.paths.dse_figures, fig_id + "." + fig_extension)
        print("PYTHON : utils : Saved figure \'{}.{}\'".format(fig_id,fig_extension))
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    def read_yaml_file(self,file):
        with open(file) as f:
            datamap = yaml.safe_load(f)
        return datamap

    def dec2hex(self,dec_array,precision):
        if precision > 24:
            hex_data = [format(x % (1 << 32), '08x') for x in dec_array]
        elif precision > 16:
            hex_data = [format(x % (1 << 24), '06x') for x in dec_array]
        elif precision > 8:
            hex_data = [format(x % (1 << 16), '04x') for x in dec_array]
        else:
            hex_data = [format(x % (1 << 8), '02x') for x in dec_array]
        return hex_data

    def hex2dec(self,hex_string,precision):
        dec_list=[]
        for data in hex_string:
            try:
                a = int(data, 16)
            except:
                a = 2 ** (precision-1)
            if a & (1 << (precision-1)):
                a -= 1 << precision
            dec_list.append(a)
        return dec_list

    def float2fixed(self,float_num_list, int_width, fractional_width):
        fixed_num_list = []
        for float_num in float_num_list:
            total_width = int_width + fractional_width
            temp1 = float_num * 2 ** fractional_width
            if (temp1 > 0 and temp1 > 2 ** (total_width - 1) - 1):
                fixed_num = 2 ** (total_width - 1) - 1
                print('error in float2fixed conversion +ve saturation implemented')
            elif (temp1 < 0 and temp1 < 2 ** (total_width - 1)):
                fixed_num = -1 * 2 ** (total_width - 1)
                print('error in float2fixed conversion -ve saturation implemented')
            else:
                fixed_num = temp1
            fixed_num_list.append(fixed_num)
        return fixed_num_list

    def float2int(self,float_num_list, int_width, mode = 'round'):
        fixed_num_list = []
        for float_num in float_num_list:
            fixed_num = float_num * (2**(int_width-1))
            if mode == 'ceil':
                fixed_num_list.append(math.ceil(fixed_num))
            else:
                fixed_num_list.append(math.floor(fixed_num))
        return fixed_num_list

    def dec2hex(self, dec_data, precision):
        hex_digits = math.ceil(precision/4)
        if (dec_data>=0):
            string = "{0:064b}".format(dec_data)
            string = '0'*(64-precision) + string[-precision:]
        else:
            string = "{0:064b}".format(2**64 - abs(dec_data))
            string = '0'*(64-precision) + string[-precision:]
        hex_split = [string[i*4:(i+1)*4] for i in range(int(64/4))]
        hex_data = ''.join([hex(int(k,2))[2:] for k in hex_split[-hex_digits:]])
        return '0x'+hex_data


    def hex2dec(self, hex_data, precision):
        if 'x' in hex_data[:2].lower():
            hex_data = hex_data.lower().split('x')[1]
        temp_bin = bin(int(hex_data, 16))[2:]
        len_bin  = len(temp_bin)
        hex_sign = temp_bin.zfill(64)[64-precision]
        if len_bin > precision:
            dec_data = int(temp_bin[-precision:], 2)
        else:
            dec_data = int(temp_bin, 2)
        if hex_sign == '0':
            return dec_data
        else:
            return -(2**(precision) - dec_data)
            
    def print_dict(self, dict_data, columns=6, space=20, tab_size=4, sparator=' ', align=True):
        pline = ' '*tab_size + ''
        for i, key in enumerate(dict_data.keys(), start=1):
            temp = "{}={}{}".format(key, dict_data[key], sparator)
            if align:
                temp = temp + ' ' * (space - len(temp))
            else:
                temp = temp + ' ' * space
            pline += temp
            if i % columns == 0:
                pline += '\n' + ' '*tab_size
        return pline

    def plot_stacked_bar(self, data):

        width  = 0.35
        labels = list(data['ops'].keys())
        values = list(data['ops'].values())
        colors = ["#006D2C", "#31A354", "#74C476"]
        fig, ax = plt.subplots()
        #for i in data['ops'].keys():
        #    ax.bar(i, data['ops'][i], width, label='Men', stacked=True)

        margin_bottom = 0
        for i, num in enumerate(data['ops'].keys()):
            values = data['ops'][num]
            ax.bar(x='Year', y='Value', stacked=True, bottom=margin_bottom, color=colors[i], label=num, width=width)
            margin_bottom += values
        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.legend()
        plt.show()
        return

class merge_csv_files_class():
    def __init__(self):
        self.hi='hi'

    def load_csv_dataset(self, dataset_file):
        with open(dataset_file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            rawdata = [row for row in readCSV]
        return rawdata

    def save_dataset_as_csv(self, dataset_file, dataset):
        with open(dataset_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(dataset)

    def drop_bad_words_from_list(self, dataset):
        bad_words = ['sum(','max(','min(','np','mean(','.','(',')', ' ']
        dataset_header = dataset[0]
        new_dataset_header = []
        for aBadWord in bad_words:
            dataset_header = [i.replace(aBadWord, '') for i in dataset_header]

        for i in range(len(dataset[0])):
            dataset[0][i] = dataset_header[i]
        return dataset

    def add_tag_to_labels(self, dataset, tag):
        dataset_header = dataset[0]
        new_dataset_header = []
        for label in dataset_header:
            new_dataset_header.append(label+tag)

        for i in range(len(dataset[0])):
            dataset[0][i] = dataset_header[i]
        return dataset

    def run_all(self, list_of_files_list, list_of_dest_files, remove_labels):
        for i in range(len(list_of_files_list)):
            merged_list = self.merge_csv_files(list_of_files_list[i], remove_labels)
            new_dataset = self.drop_bad_words_from_list(merged_list)
            self.save_dataset_as_csv(list_of_dest_files[i], new_dataset)

    def merge_csv_files(self, files_list, remove_labels, tag_list):
        csv_data = []
        labels_count = 0
        csv_data_len = []
        for csv_file in files_list:
            fname = csv_file.split('/')[-1].split('.')[0]
            temp1 = np.array(self.load_csv_dataset(csv_file))
            csv_data_len.append(len(temp1))
            temp2 = temp1.T
            labels_count = labels_count + len(temp2)
            print('{} has {} labels with len of {}'.format(fname, len(temp2),len(temp1)))
            csv_data.append(temp2)

        print('Total number of labels are : {}\n'.format(labels_count))

        if sum(csv_data_len)/len(csv_data_len) != csv_data_len[0]:
            sys.exit('The datasets do not have the same length.')

        merged_list = []
        for i, data in enumerate(csv_data):
            for line in data:
                if line[0] not in remove_labels:
                    line[0] = line[0] + tag_list[i]
                    merged_list.append(line)

        merged_list = np.array(merged_list).T.tolist()
        return merged_list



def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-func", help="To run a function or class" , default='skip')
    options = parser.parse_args(args)
    return options

if __name__ == "__main__":

    options = getOptions()

    if options.func == 'merge_csv_files':
        files_list1 = ['../training_dataset/cpf_400_PS/Features/conv_layer_features/dnn_conv_layer_cfg_graph_data.csv',
                       '../training_dataset/cpf_400_PS/Features/conv_layer_features/dnn_conv_layer_ir_data.csv',
                       '../training_dataset/cpf_400_PS/dse_conv_3DT1_L1.csv']
        dest_file1 = '../training_dataset/cpf_400_PS/conv_feature_merged.csv'

        files_list2 = ['../training_dataset/cpf_400_PS/Features/pool_layer_features/dnn_pool_layer_cfg_graph_data.csv',
                       '../training_dataset/cpf_400_PS/Features/pool_layer_features/dnn_pool_layer_ir_data.csv',
                       '../training_dataset/cpf_400_PS/dse_ds_3DT1_L2.csv']
        dest_file2 = '../training_dataset/cpf_400_PS/pool_feature_merged.csv'

        files_list3 = ['../training_dataset/cpf_400_PS/Features/fc_layer_features/dnn_fc_layer_cfg_graph_data.csv',
                       '../training_dataset/cpf_400_PS/Features/fc_layer_features/dnn_fc_layer_ir_data.csv',
                       '../training_dataset/cpf_400_PS/dse_fc_T1_L3.csv']
        dest_file3 = '../training_dataset/cpf_400_PS/fc_feature_merged.csv'

        files_list4 = ['../training_dataset/cpf_400_PS/Features/dnn_LeNet_features/dnn_lenet_cfg_graph_data.csv',
                       '../training_dataset/cpf_400_PS/Features/dnn_LeNet_features/dnn_lenet_ir_data.csv',
                       '../training_dataset/cpf_400_PS/dse_dnn_LeNet.csv']
        dest_file4 = '../training_dataset/cpf_400_PS/dnn_feature_merged.csv'

        remove_labels = ['']  # '' is required for all
        list_of_files_list = [files_list1, files_list2, files_list3, files_list4]
        list_of_dest_files = [dest_file1, dest_file2, dest_file3, dest_file4]
        merge_csv_files_class = merge_csv_files_class()

        merge_csv_files_class.run_all(list_of_files_list, list_of_dest_files, remove_labels)

    elif options.func == 'merge_ml_csv_files':
        files_path = 'C:/Users/mxs161831/Box/HLS_runs/combined'
        dest_dirs = ['conv', 'ds', 'fc']
        labels = ['BRAM', 'DSP', 'latency', 'FF', 'LUT', 'exec us']
        remove_labels = ['']
        tag_list = []
        merge_csv_files_class = merge_csv_files_class()

        for dir in dest_dirs:
            files_list = []
            for label in labels:
                file = '{}_{}*.csv'.format(dir, label)
                csv_files = os.path.join(files_path, dir, file)
                files_list = glob.glob(csv_files)
                dest_file = 'merged_{}_{}'.format(dir, label)
                dest_file = os.path.join(files_path, dir,dest_file)
                for i in files_list:
                    tag_list.append('_'+i.split('\\') [-1].split('_')[3])


                merged_list = merge_csv_files_class.merge_csv_files(files_list, remove_labels, tag_list)
                merge_csv_files_class.save_dataset_as_csv(dest_file + '.csv', merged_list)
                df_new = pandas.read_csv(dest_file + '.csv')
                GFG = pandas.ExcelWriter(dest_file + '.xlsx', sheet_name='CONV')
                df_new.to_excel(GFG, index=False)
                GFG.save()

            #print("PYTHON : utils is imported")

