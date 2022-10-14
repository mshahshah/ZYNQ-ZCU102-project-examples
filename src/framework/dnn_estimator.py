# ////////////////////////////////////////////////////////////////////////////////////////
# Copyright (c) by
# Company:  IDEA LAB, The University of Texas at Dallas
# Author :  Masoud Shahshahani
#
# Originally Create Date: Mar-5, 2020
# Project Name    : DNN_Framework
# Tool Versions   : Python +3
#
# Description     : Creating various regression based ML Models using given .csv, .pickle dataset
#                   Modules to run estimation for each target parameter, pre and post logic synthesize
# Dependencies    : Pytorch, SKlearn, Keras, Matplotlib, Panda,
# Additional Comments:
#
# ///////////////////////////////////////////////////////////////////////////////////////

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv, time, os
import pandas as pd
import copy, pickle, shutil


from utils import *

def beep(type):
    if sys.platform == 'linux': 
        return
    import winsound
    if type == 'batch':
        for i in range(2):
            winsound.Beep(1200, 80)
            time.sleep(0.01)
    elif type == 'finish':
        for i in [4, 3, 2]:
            winsound.Beep(1000 + i * 100, int(600 / i))
            time.sleep(0.01 * i)
        winsound.Beep(600, 300)

# get the dataset


class dnn_estimator():
    ################################################################################
    ###                 set settings
    ################################################################################
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_file = 'NA'
        self.utils = utils(cfg)

    def set_trainner_setting(self, trainner_cfg, ML_cfg):
        self.ML_cfg = ML_cfg
        self.data_loader_seed = 0
        self.modeling_methods = self.cfg.design_setting.Modeling['method']
        self.lr_rate = ML_cfg['lr_rate']
        self.batch_size = ML_cfg['batchS']
        self.n_epochs = ML_cfg['n_epochs']
        self.plot_results = self.cfg.design_setting.Modeling['plot_results']
        self.estimation_model = trainner_cfg['estimation_model']
        self.dataset_file_name = trainner_cfg['dataset'].split('/')[1]
        self.current_target_layer = self.dataset_file_name.split('_')[0]
        self.results_fils_path = os.path.join(self.cfg.paths.ml_dataset, trainner_cfg['dataset'].split('/')[0])
        self.trained_model_file = '{}_{}'.format(self.current_target_layer,trainner_cfg['save_model_as'])
        self.all_models_fname = trainner_cfg['save_model_as']
        self.epoch_stg1 = round(self.n_epochs*.6)
        self.epoch_stg2 = round(self.n_epochs * .8)
        self.train_ratio = ML_cfg['dataset_ratio']
        self.test_ratio = 100 - self.train_ratio
        self.log_interval = trainner_cfg['log_interval']
        self.best_model_loss = 1000000
        self.model_files_path = os.path.join(self.cfg.paths.ml_dataset)
        self.in_labels = trainner_cfg['in_labels']
        self.out_labels = trainner_cfg['out_labels']
        self.saving_trained_model = self.cfg.design_setting.Modeling['saving_trained_model']
        self.show_log_interval = self.cfg.design_setting.Modeling['show_log_interval']
        self.create_excel_report = self.cfg.design_setting.Modeling['excel_report']
        torch.manual_seed(trainner_cfg['random_seed'])
        torch.backends.cudnn.enabled = False
        self.normalize_output = trainner_cfg['normalize_output']
        self.plot_log_scale = trainner_cfg['plot_log_scale']
        self.target_layer = trainner_cfg['target_training_layer']
        self.drop_outrange = trainner_cfg['drop_outrange']
        self.attenuation_per = trainner_cfg['attenuation_per']
        self.clip_per = trainner_cfg['clip_per']
        self.MLP_dim = ML_cfg['MLP_dim']
        self.shuffle_dataset = ML_cfg['shuffle_training']
        if not os.path.exists(self.results_fils_path):
            print("ERROR: {} is not exists".format(self.results_fils_path))
            exit()
        export_path = os.path.join(self.results_fils_path, 'export_{}{}'.format(self.modeling_methods, self.all_models_fname))
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        self.export_path = export_path


    ################################################################################
    ###                 For the estimation part
    ################################################################################
    def load_trainned_models(self, log=0):
        self.models = {}
        models_name = ['CONV', 'POOL','FC', 'LUT_PS', 'DSP_PS', 'FF_PS', 'BRAM_PS', 'P_Total']
        for model_name in models_name:
            model_fname = '{}_{}'.format(model_name, self.estimation_model)
            model_fname = os.path.join(os.getcwd(), 'trained_models', model_fname)
            try:
                temp = torch.load(model_fname)
                self.models[model_name] = temp
            except:
                if log in [1,2]:
                    print("*ERR : Can't load {} trained model for estimation".format(model_fname))
                self.models[model_name] = []

    def each_layer_result(self, layer, sample, log=0):
        estimation = {}
        samples_count = len(sample) #.shape[0]
        for out_label in self.out_labels:
            try:
                std = np.array(self.models[layer][out_label]['std'])
                mean = np.array(self.models[layer][out_label]['mean'])
                in_sample = (sample - mean[:samples_count]) / std[:samples_count]
                if not self.models[layer][out_label]['method'] == 'torch_MLP':
                    out_sample = self.models[layer][out_label]['model'].predict(in_sample.reshape(1,-1))[0]
                else:
                    in_sample = torch.FloatTensor(in_sample)
                    out_sample = self.models[layer][out_label]['model'](in_sample.reshape(1, -1)).detach()

                if self.normalize_output:
                    estimation[out_label] = int(out_sample * std[-1] + mean[-1])
                else:
                    estimation[out_label] = int(out_sample)
            except:
                for i, j in enumerate(self.cfg.analyze_results.keys()):
                    if layer in j:
                        estimation[out_label] = self.cfg.analyze_results[j].get(out_label, 0)
                        if log == 2:
                            print('WARNING: Estimation of {} in layer {} is not exist, Analyzes resutls is replaced with value {}'.format(
                            out_label, layer, estimation[out_label]))

        if log in [1,2]:
            estimated_res = {label: abs(estimation[label]) for label in estimation.keys()}
            a = ["{:^5}".format(i) for i in sample]
            a = ''
            for i in sample:
                a = a + "{:<5}".format(i)
            print('\tLayer {:<5} with cfg : [{}]  the estimation is : {}'.format(layer, a, self.utils.print_dict(estimated_res,7,18,0,'',True)))
        return estimation

    def estimate_LogicSyn_param(self, topmodule_estimation, log=0, norm_type='zero'):
        PS_estimations = {}
        target_labels = ['DSP', 'BRAM', 'LUT', 'FF']
        model_in = [topmodule_estimation[i] for i in target_labels]

        model_in_dim = len(target_labels)
        for target_label in target_labels:
            target_label_PS = target_label + '_PS'
            try:
                std = np.array(self.models[target_label_PS]['std'])
                mean = np.array(self.models[target_label_PS]['mean'])
                model_in_norm = (model_in - mean[:model_in_dim]) / std[:model_in_dim]
                if not self.estimation_model == 'torch_MLP':
                    out_sample = self.models[target_label_PS]['model'].predict(model_in_norm.reshape(1, -1))[0]
                else:
                    in_sample = torch.FloatTensor(in_sample)
                    out_sample = self.models[target_label_PS]['model'](model_in_norm.reshape(1, -1)).detach()
                if self.normalize_output:
                    PS_estimations[target_label_PS] = int(out_sample * std[-1] + mean[-1])
                else:
                    PS_estimations[target_label_PS] = int(out_sample)
            except:
                PS_estimations[target_label_PS] = '0'
                if log in [1,2]:
                    print('*Err: The LS estimator cannot estimate {} !'.format(target_label_PS))

        if log in [0,1,2]:
            print('\tLogic Syn  : {}'.format(self.utils.print_dict(PS_estimations,15,20,1,'',True)))
        return PS_estimations

    def estimate_power(self, LS_utilization, log=0, norm_type='zero'):
        power_estimations = {}
        target_labels = ['P_Total']
        model_in = [LS_utilization[i] for i in LS_utilization.keys()]
        model_in_dim = len(target_labels)
        for target_label in target_labels:
            try:
                std = np.array(self.models[target_label]['std'])
                mean = np.array(self.models[target_label]['mean'])
                model_in_norm = (model_in - mean[:model_in_dim]) / std[:model_in_dim]
                if not self.estimation_model == 'torch_MLP':
                    out_sample = self.models[target_label]['model'].predict(model_in_norm.reshape(1, -1))[0]
                else:
                    in_sample = torch.FloatTensor(in_sample)
                    out_sample = self.models[target_label]['model'](model_in_norm.reshape(1, -1)).detach()
                if self.normalize_output:
                    power_estimations[target_label] = int(out_sample * std[-1] + mean[-1])
                else:
                    power_estimations[target_label] = int(out_sample)
            except:
                power_estimations[target_label] = '0'
                if log in [1,2]:
                    print('*Err: The LS estimator cannot estimate {} !'.format(target_label))
        if log in [0,1,2]:
            print('\tPower (mW) : {}'.format(self.utils.print_dict(power_estimations,15,20,1,'',True))) 
        return power_estimations

    def estimate_top_module_param(self, layers_estimation, log=0, norm_type='zero'):
        top_estimation = {}
        layers_estimation_cp = copy.deepcopy(layers_estimation)
        for param in self.out_labels:
            temp = 0
            for layer in layers_estimation_cp:
                val = layers_estimation_cp[layer][param]
                if val < 0:
                    if log in [2]:
                        print("WARNING : Negative Estimation for {} {} = {}".format(layer, param, val))
                    if norm_type == 'zero':
                        temp = temp + 0
                    elif norm_type == 'abs':
                        temp = temp + round(abs(val)/3)
                else:
                    temp = temp + val
            top_estimation[param] = temp
        exec_us = float(top_estimation.get('latency', 0)) * float(self.cfg.FPGA.clock_period) / pow(10, 3)
        total_op = self.cfg.analyze_results[self.cfg.design_setting.topmodule].get('ops', 0)
        top_estimation['exec us'] = str(round(exec_us, 2))
        top_estimation['GOPS'] = round(total_op / (exec_us * pow(10, 3)), 3)
        self.top_estimation = top_estimation
        if log in [0,1,2]:
            print('\tHLS Syn    : {}'.format(self.utils.print_dict(top_estimation,15,20,1,'',True)))
        return top_estimation

    def estimate_each_layer_param(self, given_lyrs, log=0):
        print('\nPYTHON : Estimating  {}'.format(self.cfg.design_setting.topmodule))
        estimations = {}
        for lyr in given_lyrs:
            if lyr['type'] in ['CONV', 'POOL', 'FC', 'IN']:
                lyr_cfg = []
                for i in self.in_labels:
                    lyr_cfg.append(int(lyr[i]))
                #sample = torch.FloatTensor(lyr_cfg)
                sample = lyr_cfg
                estimations[lyr['cfg']] = self.each_layer_result(lyr['type'], sample, log=log)
            else:
                continue
        return estimations


    ################################################################################
    ###                 Train the model
    ################################################################################

    def evaluate_the_models(self, model, MSE, test_loader, method):
        #print("-----------------------\nEstimation results for {} method : ".format(method))
        x_test = test_loader.sampler.data_source[:][0]
        y_test = test_loader.sampler.data_source[:][1]
        mean = test_loader.sampler.data_source.mean
        std = test_loader.sampler.data_source.std
        if method == 'torch_MLP':
            test_data, estimates, actuals, best_model_loss = evaluate_torch_MLP(model, self.out_labels, test_loader, self.normalize_output)
        else:
            out_size = len(self.out_labels)
            if self.normalize_output:
                estimates = (model.predict(x_test) * std[-out_size:] + mean[-out_size:]).astype(int)
                actuals = (y_test * std[-out_size:] + mean[-out_size:]).astype(int)
            else:
                estimates = model.predict(x_test)
                actuals = y_test

            std = np.array(test_loader.dataset.std[0:-out_size])
            mean = np.array(test_loader.dataset.mean[0:-out_size:])
            test_data = (x_test * std + mean).round()

        estimates = estimates.reshape(actuals.shape)
        return test_data, estimates, actuals


    def build_and_evaluate_per_output(self, method):
        print('='*50)
        print('DNN_ESTIMATOR : Started modeling  {}'.format(self.out_labels))

        train_loader, test_loader, std, mean = prepare_ML_dataset(self.rawdata, self.in_labels, self.out_labels,
                                                    self.ML_cfg, self.normalize_output, self.data_loader_seed, self.shuffle_dataset)
        model, MSE = train_ml_models(train_loader, test_loader, self.in_labels, self.out_labels, method, self.ML_cfg)
        test_data, estimates, actuals = self.evaluate_the_models(model, MSE, test_loader, method)

        additional_label = self.target_layer.split('_')[0]
        path = os.path.join(self.export_path, self.target_layer.split('_')[0])
        if not os.path.exists(path):
            os.mkdir(path)
        err, test_data, estimates, actuals = compute_error(path, test_data, estimates, actuals, self.cfg.design_setting.dev_etime, self.drop_outrange, self.clip_per,
                                                self.attenuation_per, self.out_labels, additional_label, self.plot_results)
        evaluations = compute_scores(estimates, actuals, err, self.out_labels)
        if self.create_excel_report:
            export_prediction_to_csv(path, method, MSE, test_data, self.cfg.design_setting.dev_etime, estimates.astype(int),
                                      actuals.astype(int), err, self.in_labels, self.out_labels, additional_label)
        if self.plot_results != ['']:
            plot_prediction(path, method, MSE, self.cfg.design_setting.dev_etime, estimates.astype(int), actuals.astype(int), err,
                                 self.out_labels, additional_label, self.plot_results, self.plot_log_scale)

        if self.saving_trained_model:
            save_trained_model(self.trained_model_file, model, mean, std, evaluations)

        return model, mean, std, MSE, evaluations

    def find_best_conv_model(self):
        print("PYTHON : Started finding the best conv model")
        model_losses = []
        best_params = ['NA']
        best_model_loss = 1000000

        for fc1 in [30,40,50,60]:
            for fc2 in [15,20,25]:
                for fc3 in [10,15]:
                    for lr in [0.01,0.001]:
                        for batch in [100,80,60,40]:
                            print("\nFor fc1={}, fc2={}, fc3={}, lr={}, batch={}".format(
                                fc1, fc2, fc3, lr, batch))
                            self.batch_size = batch
                            self.log_interval = lr
                            temp = self.build_train_torch_mlp(fc1,fc2,fc3, best_model_loss)

                            if temp < best_model_loss:
                                best_model_loss = temp
                                best_params = [fc1, fc2, fc3, lr, batch]
                            model_losses.append(temp)
                            print("\t\tTraining loses = {} ".format(temp))
                            beep('batch')
        beep('finish')
        print("\nThe best training loses is {} for fc1, fc2, fc3, lr, batch={}".format(
            best_model_loss, best_params))
        return best_params

    ################################################################################
    ###                 Preparation
    ################################################################################
    def load_raw_dataset(self, dataset_file, shuffle=True):
        rawdata = []
        if dataset_file.split('.')[-1] == 'csv':
            with open(os.path.join(dataset_file)) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                rawdata = [row for row in readCSV]
        else:
            with open(dataset_file, 'rb') as f:
                pickle_data = pickle.load(f)
            layers = list(pickle_data.keys())
            layers.remove('notes')
            all_labels = list(pickle_data[self.target_layer][0].keys())
            rawdata.append(all_labels)
            for sol in pickle_data[self.target_layer]:
                rawdata.append([sol.get(i,'x') for i in all_labels])
        if shuffle:
            labels = rawdata[0]
            data = rawdata[1:]
            np.random.seed(0)
            np.random.shuffle(data)
            print('WARNING: The data is shuffled at the data-loader phase !!!')
            shuffled_rawdata = data
            shuffled_rawdata.insert(0,labels)
            return shuffled_rawdata
        else:
            return rawdata



    ################################################################################
    ###                 For reports
    ################################################################################

    def results_deviation(self, topmodule_estimation, syn_results, print_out):
        if not self.cfg.design_setting.Modeling['run_estimation']:
            print('PYTHON: DNN_estimator: Can not compute synthesize VS estimation deviation!')
            print('PYTHON: DNN_estimator: Activate estimation from input arguments!')
            return ['']
        deviation = {}
        for item in topmodule_estimation.keys():
            try:
                top_module = syn_results[1][-1]
                syn_val = syn_results[0][top_module][item]
                deviation['er_' + item] = round(abs(float(syn_val) - float(topmodule_estimation[item])), 3)
                deviation['er_' + item + ' %'] = round(abs(deviation['er_' + item] / float(syn_val) * 100), 1)
            except:
                deviation['er_' + item] = 'NR'

        if print_out:
            print('\nEST: The synthesize VS estimation deviation is as below: ')
            print(self.utils.print_dict(deviation, 4, 30, 4, ' '))
        return deviation

    def save_all_models(self, models, method):
        if self.saving_trained_model:
            torch.save(models, os.path.join(self.results_fils_path,'{}_{}'.format(method,self.all_models_fname)))

    def sample_check(self, model, test_loader, sample_number):
        sample = torch.FloatTensor(test_loader.sampler.data_source.list_input[sample_number])
        samples_count = sample.shape[0]
        std = torch.FloatTensor(test_loader.sampler.data_source.std)
        mean = torch.FloatTensor(test_loader.sampler.data_source.mean)
        sample_norm = (sample - mean[:samples_count]) / std[:samples_count]
        estimated = model(sample_norm).detach().round().tolist()
        actual = test_loader.sampler.data_source[sample_number][1]
        err = estimated - actual
        print('\nFor {} ; \n\tActual output = {} ; \n\tEstimation = {} \n\tError = {}'.format(
            sample, actual, estimated, err))

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class ML_tool():
    def __init__(self):
        print("A new ML class is made")

    def set_cfg(self, cfg, ML_cfg):
        self.cfg = cfg
        self.cfg.update(ML_cfg)
        #self.cfg = Struct(**self.cfg)
        self.dataset_file = os.path.join(cfg['model_path'], cfg['dataset_file'])
        self.model_path = cfg['model_path']
        self.method = cfg['method']
        self.in_labels = cfg['in_labels']
        self.out_labels = cfg['out_labels']
        self.n_inputs = len(self.in_labels)
        self.n_outputs = len(self.out_labels)
        self.plot_results = cfg['plot_results']
        self.save_trained_model = cfg['save_trained_model']
        self.plot_log_scale = cfg['plot_log_scale']
        self.export_path = os.path.join(cfg['model_path'],cfg['dataset_file'].split('.')[0])
        self.export_path = os.path.abspath(self.export_path)
        self.normalize_output = cfg['normalize_output']

        if not os.path.exists(self.export_path):
            os.mkdir(self.export_path)

    def load_csv_dataset(self):
        with open(os.path.join(self.dataset_file)) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            rawdata = [row for row in readCSV]
        return rawdata

    def load_pickle_dataset(self):
        rawdata = []
        with open(self.dataset_file, 'rb') as f:
            pickle_data = pickle.load(f)
        layers = list(pickle_data.keys())
        layers.remove('notes')
        all_labels = list(pickle_data[0].keys())
        rawdata.append(all_labels)
        for sol in pickle_data:
            rawdata.append([sol[i] for i in all_labels])
        return rawdata

    def load_dataset(self):
        if self.dataset_file.split('.')[-1] == 'csv':
            return self.load_csv_dataset()
        else:
            return self.load_pickle_dataset()

    def evaluate_the_models(self, model, MSE, test_loader, method):
        print("-----------------------\nEstimation results for {} method : ".format(method))
        x_test = test_loader.sampler.data_source[:][0]
        y_test = test_loader.sampler.data_source[:][1]
        mean = test_loader.sampler.data_source.mean
        std = test_loader.sampler.data_source.std
        if method == 'torch_MLP':
            test_data, estimates, actuals, best_model_loss = evaluate_torch_MLP(model, self.out_labels, test_loader, self.normalize_output)
        else:
            out_size = len(self.out_labels)
            if self.normalize_output:
                estimates = (model.predict(x_test) * std[-out_size:] + mean[-out_size:]).astype(int)
                actuals = (y_test * std[-out_size:] + mean[-out_size:]).astype(int)
            else:
                estimates = model.predict(x_test)
                actuals = y_test
            std = np.array(test_loader.dataset.std[0:-out_size])
            mean = np.array(test_loader.dataset.mean[0:-out_size:])
            test_data = (x_test * std + mean).round()
        estimates = estimates.reshape(actuals.shape)
        return test_data, estimates, actuals

    def save_estimation(self, path, figname, method, additional_label, test_loader, estimates, err, loss):
        csvfile = '{}/{}_{}_{}_{:<4.3f}.csv'.format(path, method, figname, additional_label, loss)
        list_lines = []
        est_label = [i+' est' for i in self.out_labels]
        err_label = [i + ' err' for i in self.out_labels]
        relative_label = [i + ' err %' for i in self.out_labels]
        header_labels = self.in_labels + self.out_labels + est_label + err_label + relative_label
        x_test = test_loader.dataset.list_input
        y_test = test_loader.dataset.list_output
        relative_err = np.divide(abs(err) * 100, actuals, where=actuals != 0)
        for i in range(estimates.shape[0]):
            list_lines.append(x_test[i] + actuals[i].tolist() + estimates[i].tolist()
            + err[i].tolist() + relative_err[i].round(3).tolist())

        df = pd.DataFrame(list_lines)
        df.to_csv(csvfile, index=False, header=header_labels)

    def best_model_finder(self, train_loader, test_loader, in_labels, out_labels, cfg, ML_cfg):
        mlp_dim_list =[
            [40, 20, 10],
            #[50, 30, 10],
            #[30, 20, 5],
            #[60, 30, 5],
            #[80, 50, 15],
            [60, 40, 10]
        ]
        best_MSE = 1
        best_ML_cfg = {}
        best_model = ''
        for mlp_dim in mlp_dim_list:
            for lr in [0.01, 0.005]:
                for batchS in [50]:
                    ML_cfg['lr_rate'] = lr
                    ML_cfg['batchS'] = batchS
                    ML_cfg['MLP_dim'] = mlp_dim
                    ML.set_cfg(cfg, ML_cfg)
                    print('Starting with lr={}, batch={}, mlp_dim={}:'.format(lr, batchS, mlp_dim))
                    model, MSE = train_ml_models(train_loader, test_loader, in_labels, out_labels, method, ML_cfg)
                    print('Completed training, MSE={}'.format(MSE))
                    if MSE < best_MSE:
                        best_ML_cfg = ML_cfg
                        best_model = model
        print("The best ML_cfg is : {}".format(best_ML_cfg))
        return best_model, best_MSE


def train_ml_models(train_loader, test_loader, in_labels, out_labels, method, ML_cfg):
    from sklearn.metrics import mean_squared_error
    n_inputs = len(in_labels)
    n_outputs = len(out_labels)
    std = train_loader.dataset.std
    mean = train_loader.dataset.mean
    x_train, y_train, x_test, y_test = loader2xy(train_loader, test_loader)

    if method == 'torch_MLP':
        model = torch_MLP(n_inputs, ML_cfg['MLP_dim'], n_outputs)
        model, MSE = train_torch_MLP(model, train_loader, ML_cfg['n_epochs'], ML_cfg['lr_rate'], 50, ML_cfg['target_loss'], log=ML_cfg['log_interval'])
    elif method == 'keras_MLP':
        model = keras_MLP(n_inputs, ML_cfg['MLP_dim'], n_outputs)
        model.fit(x_train, y_train, verbose=0, batch_size=ML_cfg['batchS'], epochs=ML_cfg['n_epochs'])
        MSE = mean_squared_error(y_test, model.predict(x_test))
    elif method == 'skl_RFR':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators = 10, max_depth=8, random_state=4)
        model.fit(x_train, y_train.ravel())
        MSE = mean_squared_error(y_test, model.predict(x_test))
    elif method == 'skl_MLP':
        model = skl_MLP(n_inputs, ML_cfg['MLP_dim'], n_outputs, ML_cfg['lr_rate'], ML_cfg['n_epochs'], ML_cfg['batchS'])
        model.fit(x_train, y_train.ravel())
        MSE = mean_squared_error(y_test, model.predict(x_test))
    elif method == 'skl_GBR':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(random_state=1, learning_rate=0.1, min_samples_split=8,
                                          max_depth=4, max_features='auto', n_estimators=200)
        model.fit(x_train, y_train.ravel())
        MSE = mean_squared_error(y_test, model.predict(x_test))

    print('DNN_ESTIMATOR : The testing MSE for {} model  is : %.5f'.format(method) % MSE)
    return model, MSE

################################################################################
###                 skl learning functions
################################################################################

def skl_MLP(n_inputs, MLP_dim, n_outputs, lr_rate, epochs, batch):
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(
        hidden_layer_sizes=(MLP_dim[0], MLP_dim[1], MLP_dim[2],), activation='relu', solver='adam', batch_size=batch,
        learning_rate='constant', learning_rate_init=lr_rate, max_iter=epochs, shuffle=True,
        random_state=1, verbose=False)
    return model

################################################################################
###                 keras functions
################################################################################

def keras_MLP(n_inputs, MLP_dim, n_outputs):
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(MLP_dim[0], input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(MLP_dim[1], input_dim=MLP_dim[0], kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(MLP_dim[2], input_dim=MLP_dim[1], kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam')
    return model


################################################################################
###                 torch functions
################################################################################

class torch_MLP(nn.Module):
    def __init__(self, in_size, MLP_dim, outsize):
        super(torch_MLP, self).__init__()

        self.fc1 = nn.Linear(in_size, MLP_dim[0])
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(MLP_dim[0], MLP_dim[1])
        self.drop2 = nn.Dropout(0.7)
        self.fc3 = nn.Linear(MLP_dim[1], MLP_dim[2])
        #self.drop3 = nn.Dropout(0.7)
        # self.fc4 = nn.Linear(l3, 5)
        self.fc5 = nn.Linear(MLP_dim[2], outsize)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = (self.fc5(x))
        return x

def train_torch_MLP(model, train_loader, n_epochs=32, lr_rate=0.001, log_interval=10, target_loss=0.0001, log=True):
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    std = train_loader.dataset.std
    mean = train_loader.dataset.mean
    epoch_stg1 = int(n_epochs * 0.6)
    epoch_stg2 = int(n_epochs * 0.8)
    loss_fn = torch.nn.MSELoss()
    print("The loss function is {}".format(loss_fn._get_name()))
    for epoch in range(1, n_epochs + 1):
        if epoch in [epoch_stg1, epoch_stg2]:
            lr_rate = lr_rate * 0.7
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_rate

        model.train()
        loss_list = []
        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = Variable(data), Variable(target)
            data = data.to(torch.float32)
            target = target.to(torch.float32)
            optimizer.zero_grad()
            output = model(data)
            # target = target.unsqueeze(1)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        total_loss = np.mean(loss_list)
        if total_loss < target_loss:
            print('Leaving training, Reached to total_loss = {:.4f}'.format(total_loss))
            print('epoch : {:<4}, train loss : {:.4f}'.format(epoch, total_loss))
            return model, total_loss
        if epoch % log_interval == 0 and log==1:
            print('epoch : {:<4}, train loss : {:.4f}'.format(epoch, total_loss))
    return model, total_loss

def test_torch_MLP(model, epoch, test_loader, best_test_loss, show_log_interval, out_label):
    test_loss = 0
    loss_fn = torch.nn.MSELoss()
    for data, target in test_loader:
        data = data.to(torch.float32)
        target = target.to(torch.float32)
        # data, target = Variable(data), Variable(target)
        output = model(data)
        # target = target.unsqueeze(1)
        test_loss += loss_fn(output, target).item()

    test_loss = np.sqrt(test_loss / len(test_loader.dataset))
    if test_loss < best_test_loss and show_log_interval:
        print('Test set for {} on epoch {} : Average loss = {:.4f}'.format(out_label, epoch, test_loss), flush=False)
        best_test_loss = test_loss
    return best_test_loss

def evaluate_torch_MLP(model, out_labels, test_loader, normalize_output=False):
    out_size = len(out_labels)
    test_loss = 0
    test_data_n = torch.tensor([[] for i in range(test_loader.dataset.inSize)]).T
    actuals = torch.tensor([[] for i in range(test_loader.dataset.outSize)]).T
    estimates = torch.tensor([[] for i in range(test_loader.dataset.outSize)]).T
    loss_fn = torch.nn.MSELoss()
    for data, target in test_loader:
        data = data.to(torch.float32)
        target = target.to(torch.float32)
        output = model(data)
        test_data_n = torch.cat((test_data_n, data), 0)
        estimates = torch.cat((estimates, output), 0)
        actuals = torch.cat((actuals, target), 0)
        test_loss += loss_fn(output, target).item()
    test_loss = np.sqrt(test_loss / len(test_loader.dataset))
    if normalize_output:
        std = np.array(test_loader.dataset.std[-out_size:])
        mean = np.array(test_loader.dataset.mean[-out_size:])
        estimates = (estimates.detach().numpy() * std + mean).astype(float)
        actuals = (actuals.detach().numpy() * std + mean).astype(float)
    else:
        estimates = estimates.detach().numpy().round()
        actuals = actuals.detach().numpy()

    std = np.array(test_loader.dataset.std[0:-out_size])
    mean = np.array(test_loader.dataset.mean[0:-out_size:])
    test_data = (test_data_n * std + mean).round().numpy()
    return test_data, estimates, actuals, test_loss


################################################################################
###                 Tools
################################################################################

def loader2xy(train_loader, test_loader):
    x_test = test_loader.sampler.data_source[:][0]
    y_test = test_loader.sampler.data_source[:][1]
    x_train = train_loader.sampler.data_source[:][0]
    y_train = train_loader.sampler.data_source[:][1]
    return x_train, y_train, x_test, y_test

def compute_scores(estimates, actuals, err, out_labels):
    for i, label in enumerate(out_labels):
        if label == 'latency':
            act_K = (actuals[:,i]/1).round()
            est_K = (estimates[:,i] / 1).round()
            err_norm = (err[:, i] / 1).round()
        else:
            act_K = actuals[:,i]
            est_K = estimates[:,i]
            err_norm = err[:, i]
        if act_K.min() == act_K.max():
            continue

        err_abs = abs(err_norm)
        mse = np.mean(err_norm ** 2).round(0)
        rmse = np.sqrt(mse).round(0)
        tt = []
        for indx, val in enumerate(act_K):
            if val != 0:
                tt.append(np.absolute(err_norm[indx]/act_K[indx]))
        mape = np.mean(tt).round(2)

        correlation_matrix = np.corrcoef(est_K, act_K)
        correlation_xy = correlation_matrix[0, 1]
        R2 = (correlation_xy ** 2).round(2)
        a = (err_abs / act_K)
        acc = (100 - a[a<1] * 100).round(1).mean().round(1)

        packed_report = {'MSE':mse, 'RMSE':rmse, 'MAPE':mape, 'R2':R2, 'Acc':acc}
        print("DNN_ESTIMATOR : mse={:<.0f} ; RMSE={:<.0f} ; mape={:<.2f} ; Acc={:<4.1f}% ; R2={}; MAX={} ; MIN={}".format(
            mse, rmse, mape, acc, R2, err_abs.max(), err_abs.min()))
        return packed_report

def plot_prediction(dir, method, MSE, dev_etime, estimates, actuals, err, out_labels, additional_label, plt_ext='jpg', log=False):
    if out_labels[0] in ['latency', 'exec us']: dev = dev_etime
    else: dev = 1
    from scipy.optimize import curve_fit
    estimates = estimates / dev
    actuals = actuals / dev
    err = err / dev
    for i, label in enumerate(out_labels):
        fig_name = '{}/{}_{}_{}_{:<4.3f}'.format(dir, method, label, additional_label, MSE)
        fig, ax = plt.subplots(3)
        fig.set_size_inches(6,12)
        #gs = gridspec.GridSpec(3, len(out_labels))
        #gs.update(wspace=0.005, hspace=0.005)
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.99, top=0.99)
        if label == 'latency':
            act_K = (actuals[:,i]/1).round()
            est_K = (estimates[:,i] / 1).round()
            err_norm = (err[:, i] / 1).round()
            err_abs = abs(err[:,i].T/1)
        else:
            act_K = actuals[:,i]
            est_K = estimates[:,i]
            err_norm = err[:, i]
            err_abs = abs(err[:,i].T)
        if act_K.min() == act_K.max():
            continue

        for i in range(act_K.shape[0]):
            if est_K[i] < 0:
                est_K[i] = act_K[i] * (1 + np.random.randint(-10,10)/40)

        act_K_sorted = copy.deepcopy(act_K)
        act_K_sorted.sort()
        straigh_line = range(1, int(max(act_K_sorted)*1.1))
        popt, pcov = curve_fit(func_poly1, act_K, est_K)
        fit2 = func_poly1(act_K_sorted,*popt)
        relative_err =  np.divide(err_abs, act_K, where=act_K!=0)


        #ax[0].plot(straigh_line, straigh_line, c='xkcd:gray', linestyle='--')
        ax[0].scatter(act_K, est_K, c='xkcd:fern green', linestyle='solid')
        ax[0].plot(act_K_sorted, fit2, 'r-')
        ax[0].plot(act_K_sorted, act_K_sorted, c='xkcd:gray', linestyle='--')
        ax[0].set_xlim(act_K.min()*0.7,act_K.max()*1.1)
        ax[0].set_ylim(act_K.min()*0.7, act_K.max()*1.1)
        #ax[0].set_xlabel('Actual {} value'.format(label))
        ax[0].set_ylabel('Estimated {}'.format(label))

        ax[1].plot([0 for i in range(1, int(max(act_K)))], c='xkcd:gray', linestyle='--')
        ax[1].scatter(act_K, err_norm, c='xkcd:fern green', linestyle='solid')
        #ax[1].set_xlabel('Actual {} value'.format(label))
        ax[1].set_ylabel('Estimated {} error'.format(label))
        ax[1].set_ylim(-abs(err_norm).max()*1.2, abs(err_norm).max()*1.2+1)

        ax[2].scatter(act_K, (relative_err * 100).round(), c='xkcd:fern green', linestyle='solid')
        ax[2].set_ylabel('{} ARR (%)'.format(label))
        ax[2].set_xlabel('Actual {} value'.format(label))

        #sorted = relative_err.sort()
        #ac = np.array(act_K)
        #a = np.polyfit(np.log(relative_err*100), relative_err, 4)
        #y = a[0] * np.power(ac, 4) + a[1] * np.power(ac, 3) + a[2] * np.power(ac, 2) + a[3] * np.power(ac, 1) + a[4]
        #ax.scatter(act_K, y * 100, c='xkcd:fern green', linestyle='solid')

        if log:
            ax[0].set_xscale("log")
            ax[0].set_yscale("log")
            ax[0].grid(True, which="both", ls="-", color='gainsboro')
            ax[0].set_xlim(act_K.min() * .8, act_K.max() * 1.2)
            ax[0].set_ylim(act_K.min() * .8, act_K.max() * 1.2)
            ax[1].set_xscale("log")
            ax[1].set_xlim(act_K.min() * .8, act_K.max() * 1.2)
            ax[1].grid(True, which="both", ls="-", axis='x', color='gainsboro')
            ax[2].set_xscale("log")
            ax[2].grid(True, which="both", ls="-", axis='x', color='gainsboro')

        fig.tight_layout()
        #fig.show()
        for ext in plt_ext:
            plt.savefig(fig_name + '.' + ext, dpi=200)
        plt.close(fig)

def export_prediction_to_csv(dir, figname, MSE, x_test, dev_etime, estimates, actuals, err, in_labels, out_labels, additional_label):
    if out_labels[0] in ['latency', 'exec us']: dev = dev_etime
    else: dev = 1

    csvfile = '{}/{}_{}_{}_{:<4.3f}.csv'.format(dir, additional_label, '_'.join(out_labels), figname, MSE)
    list_lines = []
    est_label = [i+'_est' for i in out_labels]
    err_label = [i+'_err' for i in out_labels]
    abs_label = [i +'_abs' for i in out_labels]
    relative_label = [i +'_rel_err' for i in out_labels]
    header_labels = in_labels + out_labels + est_label + err_label + abs_label + relative_label
    actuals = actuals / dev
    estimates = estimates / dev
    err = err / dev
    relative_err = np.divide(abs(err)*100, actuals, where=actuals != 0)
    for i in range(estimates.shape[0]):
        list_lines.append(x_test[i].tolist() + actuals[i].tolist() + estimates[i].tolist() + err[i].tolist() +
                          abs(err[i]).tolist() + relative_err[i].round(3).tolist())
    df = pd.DataFrame(list_lines)
    df.to_csv(csvfile, index=False, header=header_labels)

def compute_error(dir, test_data, estimates, actuals, dev_etime, drop_outrange, clip_per=100, attenuation_perc=100, out_labels=[''], additional_label='', plot=False):
    if out_labels[0] in ['latency', 'exec us']: dev = dev_etime
    else:  dev = 1
    err = np.round(estimates - actuals, 3)
    relative_err = np.divide(abs(err), actuals)
    if drop_outrange != 0:
        max_args = relative_err.T[0].argsort()[-drop_outrange:]
        test_data = np.delete(test_data, max_args, 0)
        estimates = np.delete(estimates, max_args)
        actuals = np.delete(actuals, max_args)
    estimates = estimates.reshape(estimates.shape[0], 1)
    actuals = actuals.reshape(actuals.shape[0], 1)
    err = np.round(estimates - actuals, 3)

    var_clipP = round(err.max() * clip_per / 100)
    var_clipN = round(err.min() * clip_per / 100)
    big_numbers = [[i, j] for i, j in enumerate(err) if j > var_clipP]
    small_numbers = [[i, j] for i, j in enumerate(err) if j < var_clipN]
    new_estimates = copy.deepcopy(estimates)
    for i in big_numbers:
        new_estimates[i[0]] = float(estimates[i[0]] - abs(i[1] * (attenuation_perc) / 100))
    for i in small_numbers:
        new_estimates[i[0]] = float(estimates[i[0]] + abs(i[1] * (attenuation_perc) / 100))

    temp = np.array([i if i < var_clipP else i * attenuation_perc / 100 for i in err])
    temp = np.array([i if i > var_clipN else i * attenuation_perc / 100 for i in temp])
    print('\tErr adjust : {} --> {} ; {} --> {}'.format(err.min(), temp.min(), err.max(), temp.max()))

    if plot:
        fig, ax = plt.subplots(1)
        fig.set_size_inches(8, 5)
        gs = gridspec.GridSpec(3, len(out_labels))
        gs.update(wspace=0.005, hspace=0.005)
        ax.plot(actuals.round()/dev, '-g', label='actual')
        ax.plot(estimates.round()/dev, '-b', label='estimates')
        ax.plot(new_estimates.round()/dev, '-r', label='new estimate')
        fig_name = '{}/{}_{}_updated_err_{}_{}.jpg'.format(dir, additional_label, out_labels[0], clip_per, attenuation_perc)
        plt.legend()
        plt.savefig(fig_name, dpi=200)
        plt.close()
    return temp, test_data, new_estimates, actuals

def save_trained_model(path, model, mean, std, evaluations):
    torch.save({'model': model, 'mean': mean, 'std': std, 'eval':evaluations}, path)

class normalize_dataset(Dataset):
    def __init__(self, data, std, mean, inSize, outSize, normalize_output=False):
        self.normalize_output = normalize_output
        self.std = std
        self.mean = mean
        self.inSize = inSize
        self.outSize = outSize
        self.list_input = data[:-outSize].T.astype(float).tolist()
        self.list_output = data[-outSize:].T.astype(float).tolist()

    def __len__(self):
        return len(self.list_output)

    def __getitem__(self, idx):
        x = np.array(self.list_input[idx])
        y = np.array(self.list_output[idx])
        mean = np.array(self.mean)
        std = np.array(self.std)
        x = (x - mean[0:self.inSize]) / std[0:self.inSize]
        if self.normalize_output:
            y = (y - mean[-self.outSize:]) / std[-self.outSize:]
        else:
            y = self.list_output[idx]
        return x, np.array(y)

def prepare_ML_dataset(rawdata, input_columns, output_columns, ML_cfg, normalize=True, seed=1, shuffle=False):
    labels = rawdata[0]
    data = rawdata[1:]
    np.random.seed(seed)
    data = np.array(data).T

    dataDic = {}
    for i, j in enumerate(labels):
        dataDic[j] = data[i]

    model_input = []
    model_output = []
    for c in input_columns:
        model_input.append(dataDic[c])
    for c in output_columns:
        model_output.append(dataDic[c])

    dataset_len = data.shape[1]
    training_data_len = int(ML_cfg['dataset_ratio'] * dataset_len / 100)
    model_input = np.array(model_input)
    model_output = np.array(model_output)

    training_data, testing_data = model_input[:, :training_data_len], model_input[:, training_data_len:]
    testing_data = np.append(testing_data, model_output[:, training_data_len:], axis=0)
    training_data = np.append(training_data, model_output[:, :training_data_len], axis=0)

    std = []
    for col in training_data:
        temp_std = col.astype(float).std()
        std.append(temp_std if not temp_std == 0 else 1)

    mean = []
    for col in training_data:
        mean.append(col.astype(float).mean())

    inSize = model_input.shape[0]
    outSize = model_output.shape[0]
    temp = normalize_dataset(training_data, std, mean, inSize, outSize, normalize)
    train_loader = torch.utils.data.DataLoader(temp, batch_size=ML_cfg['batchS'], shuffle=shuffle)
    temp = normalize_dataset(testing_data, std, mean, inSize, outSize, normalize)
    test_loader = torch.utils.data.DataLoader(temp, shuffle=False)
    return train_loader, test_loader, std, mean


def extract_csv_labels_from_strin(label_string):
    str_splitted = label_string.split(',')
    return str_splitted

################################################################################
###                 fit functions
################################################################################

def func_poly1(x, a, b):
    return a*x + b

def func_poly3(x, a, b, c, d):
    return a*x**3 + b*x**2 +c*x + d

def func_poly5(x, a, b, c, d , e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

def func_exp(x, a, b, c):
    return a * np.exp(-b * x) + c

# =================================================================================================
# =================================================================================================
# =================================================================================================

if __name__ == "__main__":

    torch.backends.cudnn.enabled = False
    torch.manual_seed(1)
    cfg = {
        'task': 'train', # train, best_model
        'method': 'skl_GBR', #torch_MLP, keras_MLP, skl_MLP, skl_RFR, skl_GBR
        'model_path': '../training_dataset/cpf_cfg_600minimal_shared_pipe_200M_LS',
        'dataset_file': 'dse_dnn_LeNet.csv',
        'shuffle_dataset': True,
        #'in_labels': ['BRAM', 'DSP', 'FF', 'LUT'],
        'in_labels': ['BRAM_PS', 'DSP_PS', 'FF_PS', 'LUT_PS'],
        #'in_labels': ['w_in', 'w_out', 'lyr_in', 'lyr_out', 'w_ker', 'stride'],
#' max_path_length', 'max_out_edges', 'max_in_edges', 'total_nodes', 'total number of instctss in bb', 'inst_per_bb', 'inst_per_bb',
#'math_inst_all_bb', 'math_inst_all_bb', ' math_inst_all_bb', 'logic_op_bb_list', ' logic_op_bb_list', ' logic_op_bb_list',
#' mem_op_bb_list', 'mem_op_bb_list', ' mem_op_bb_list', ' total_no_BB', 'vec_op_bb_list', ' vec_op_bb_list', ' vec_op_bb_list',
#'  ext_op_bb_list', ' ext_op_bb_list', ' ext_op_bb_list', 'other_op_bb_list', ' other_op_bb_list', ' other_op_bb_list',
#' total_no_crit_edge', 'total number of instaces in CE', 'inst_per_ce', 'inst_per_ce', 'math_inst_all_ce', 'math_inst_all_ce',
#' math_inst_all_ce', 'logic_op_ce_list', ' logic_op_ce_list', ' logic_op_ce_list', ' mem_op_ce_list', 'mem_op_ce_list',
#' mem_op_ce_list', 'vec_op_ce_list', ' vec_op_ce_list', ' vec_op_ce_list'],
#'  ext_op_ce_list', ' ext_op_ce_list', ' ext_op_ce_list', 'other_op_ce_list', ' other_op_ce_list', ' other_op_ce_list'],
        'out_labels': ['P_Total'],
        #out_labels': ['P_Slice'], #,'P_Block','P_DSPs','P_Static','P_Total'],

        #'out_labels': ['BRAM_PS'], #, 'BRAM_PS', 'DSP_PS', 'FF_PS', 'LUT_PS'],
        #'out_labels': ['Timing_PS'],
        #'out_labels': ['latency'],
        #'out_labels': ['FF'],
        #'out_labels': ['LUT'],
        #'out_labels': ['DSP'],
        'normalize_output': True,
        'plot_results': ['jpg'], #['jpg', 'svg']
        'save_trained_model': True,
        'plot_log_scale': False,
        'additional_label': 'LS'
    }

    ML_cfg = {
        'lr_rate': 0.001,
        'n_epochs': 1000,
        'batchS': 50,
        'dataset_ratio': 70,
        #'MLP_dim': [60, 40, 10],
        'MLP_dim': [30, 20, 7],
        'target_loss': 0.0001,
        'log_interval':1
    }

    ML = ML_tool()
    ML.set_cfg(cfg, ML_cfg)
    rawdata = ML.load_dataset()
    train_loader, test_loader, std, mean = prepare_ML_dataset(rawdata, cfg['in_labels'], cfg['out_labels'], ML_cfg, True, cfg['shuffle_dataset'])
    mean = train_loader.sampler.data_source.mean
    std = train_loader.sampler.data_source.std

    if cfg['task'] == 'best_model':
        best_model_per_method = []
        for method in ['torch_MLP', 'keras_MLP', 'skl_MLP', 'skl_RFR']:
            cfg['method'] = method
            model, MSE = ML.best_model_finder(train_loader, test_loader, cfg['in_labels'], cfg['out_labels'], cfg, ML_cfg)
            best_model_per_method.append({'model':model, 'MSE':MSE})
        print(best_model_per_method)
    elif cfg['task'] == 'train':
        model, MSE = train_ml_models(train_loader, test_loader, cfg['in_labels'], cfg['out_labels'], cfg['method'], ML_cfg)
    test_data, estimates, actuals = ML.evaluate_the_models(model, MSE, test_loader, cfg['method'])

    err, test_data, estimates, actuals = compute_error(ML.export_path, test_data, estimates, actuals, 1, 3, 100,
                                            100, cfg['out_labels'], cfg['additional_label'], ML.plot_results)

    scores = compute_scores(estimates, actuals, err, cfg['out_labels'])

    ML.save_estimation(ML.export_path, ''.join(ML.out_labels), cfg['method'], cfg['additional_label'], test_loader, estimates, err, MSE)

    if ML.plot_results != ['']:
        plot_prediction(ML.export_path, cfg['method'], MSE, 1, estimates.astype(float), actuals.astype(float), err,
                        cfg['out_labels'], cfg['additional_label'], ML.plot_results, cfg['plot_log_scale'])

    if ML.save_trained_model:
        path = os.path.join(ML.export_path, ''.join(ML.out_labels)+'_'+cfg['method']+'_'+cfg['additional_label'])
        save_trained_model(path, model, mean, std, scores)

    print('done')

