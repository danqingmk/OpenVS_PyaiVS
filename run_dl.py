import warnings
import numpy as np
import pandas as pd
import argparse
import random as rn
import multiprocessing as mp
import time
import os
import sys


from PyaiVS.DNN_model import para_dnn
from PyaiVS.graph_model import para_dl
import itertools

np.set_printoptions(threshold=sys.maxsize)
# Random seeding
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(123)
rn.seed(123)
warnings.filterwarnings('ignore')


def file_merge(file_df, models):
    filelist = {}
    dataset = file_df.split('/')[-1].replace('.csv', '')
    df = pd.read_csv(file_df)
    cols = list(df.columns)
    cols.remove('Smiles')
    path = './dataset/{0}/result_save'.format(dataset)
    for task in cols:
        for model in models:
            for file in os.listdir(os.path.join(os.path.join(path, task), model)):
                if 'para' in file:
                    continue
                filelist.setdefault(file, []).append(
                    os.path.join(os.path.join(os.path.join(os.path.join(path, task), model)), file))
    for rtype in filelist.keys():
        rlist = filelist[rtype]
        mer = pd.DataFrame()
        for file in rlist:
            df = pd.read_csv(file)

            mer = pd.concat([mer, df], ignore_index=True)
        mer = mer.groupby(['seed', 'FP_type', 'split_type', 'type'])[
            ['se', 'sp', 'acc', 'mcc', 'precision', 'auc_prc', 'auc_roc']].mean()
        mer = mer.reset_index()
        save_path = os.path.join(path, rtype.split('_')[1])

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        abspath = os.path.join(save_path, rtype)
        mer.to_csv(abspath, index=False)
    pass


def model_set(X, Y, split_type='random', FP_type='ECFP4', model_type='SVM', model_dir=False, file_name=None,
              task_type='cla', difftasks=None):
    if model_type == 'DNN':
        para, best = para_dnn(X, Y, split_type=split_type, FP_type=FP_type, model_dir=model_dir, file_name=file_name,
                             difftasks=difftasks)

    elif model_type == 'gat':
        para, best = para_dl(X, Y, split_type=split_type, file_name=file_name, model_name='gat', model_dir=model_dir,
                            difftasks=difftasks, task_type=task_type, device='cpu')
    elif model_type == 'attentivefp':
        para, best = para_dl(X, Y, split_type=split_type, file_name=file_name, model_name='attentivefp',
                            model_dir=model_dir, difftasks=difftasks, task_type=task_type, device='cpu')
    elif model_type == 'gcn':
        para, best = para_dl(X, Y, split_type=split_type, file_name=file_name, model_name='gcn', model_dir=model_dir,
                            difftasks=difftasks, task_type=task_type, device='cpu')
    else:
        para, best = para_dl(X, Y, split_type=split_type, file_name=file_name, model_name='mpnn', model_dir=model_dir,
                            difftasks=difftasks, task_type=task_type, device='cpu')
    return para, best


def pair_param(file_name, data, split_type, model_type, FP_type, difftasks):
    if model_type in ['DNN']:
        X = data.Smiles
        Y = data[difftasks[0]]
        model_path = 'model_save/' + model_type
        model_dir = file_name.replace(file_name.split('/')[-1], model_path)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        result_path = 'result_save/' + model_type
        result_dir = file_name.replace(file_name.split('/')[-1], result_path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        para_name, best_name = os.path.join(result_dir, '_'.join([split_type, model_type, FP_type,'para.csv'])), os.path.join(
            result_dir, '_'.join([split_type, FP_type,model_type, 'best.csv']))
        if os.path.exists(para_name):

            pass
        else:
            para, best = model_set(X, Y, split_type=split_type, model_type=model_type, model_dir=model_dir,
                                   file_name=file_name, difftasks=difftasks)
            para.to_csv(para_name, index=False)
            best.to_csv(best_name, index=False)


    else:
        X = data.Smiles
        Y = data[difftasks[0]]
        model_path = 'model_save/' + model_type
        model_dir = file_name.replace(file_name.split('/')[-1], model_path)
        result_path = 'result_save/' + model_type
        result_dir = file_name.replace(file_name.split('/')[-1], result_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        para_name, best_name = os.path.join(result_dir, '_'.join([split_type, model_type, 'para.csv'])), os.path.join(
            result_dir, '_'.join([split_type, model_type, 'best.csv']))
        para, best = model_set(X, Y, split_type=split_type, model_type=model_type, model_dir=model_dir,
                                   file_name=file_name, difftasks=difftasks)
        para.to_csv(para_name, index=False)
        best.to_csv(best_name, index=False)

        return
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='we must give this para')
    parser.add_argument('--split', default=['scaffold'], nargs='*', choices=['random', 'scaffold', 'cluster'])
    parser.add_argument('--FP', default=['MACCS'], nargs='*', choices=['ECFP4', 'MACCS', '2d-3d', 'pubchem'])
    parser.add_argument('--model', default=['SVM'], nargs='*', choices=['DNN', 'KNN', 'SVM', 'RF', 'XGB'])
    parser.add_argument('--threads', default=10, type=int)
    parser.add_argument('--mpl', default=False, type=str)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()
    return args


def main(file, split, FP, models, cpus, mpl, device):
    ml_models = [model for model in models if model in ['KNN', 'SVM', 'RF', 'XGB']]
    ml = {'split': split, 'FP': FP, 'model': ml_models}
    ml = pd.DataFrame(list(itertools.product(*ml.values())), columns=ml.keys())
    dl_models = [model for model in models if model in ['gcn', 'gat', 'mpnn', 'attentivefp']]
    dl = {'split': split, 'model': dl_models}
    dl = pd.DataFrame(list(itertools.product(*dl.values())), columns=dl.keys())
    data = pd.read_csv(file, error_bad_lines=False)
    difftasks = list(data.columns)
    difftasks.remove('Smiles')

    if mpl == False:
        for i in range(len(ml.index.values)):
            start_time = time.time()
            split_type = ml.iloc[i].split
            FP_type = ml.iloc[i].FP
            model_type = ml.iloc[i].model
            a = pair_param(file, data, split_type, model_type, FP_type, difftasks)
            use_time = round(time.time() - start_time, 5)
            if use_time >= 1:
                local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                record_save = file.replace(file.split('/')[-1], 'record_every_model.csv')
                f = open(record_save, 'a+')
                f.write(",".join([file, split_type, model_type, str(use_time), local_time, FP_type, '\n']))
                f.close()
    else:
        p = mp.Pool(processes=cpus)
        for i in range(len(ml.index.values)):
            start_time = time.time()
            split_type = ml.iloc[i].split
            FP_type = ml.iloc[i].FP
            model_type = ml.iloc[i].model
            result = p.apply_async(pair_param, args=(file, data, split_type, model_type, FP_type, difftasks))
            use_time = int(time.time() - start_time)
            if use_time >= 1:
                local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                record_save = file.replace(file.split('/')[-1], 'record_every_model.csv')
                f = open(record_save, 'a+')
                f.write(",".join([file, split_type, model_type, str(use_time), local_time, FP_type, '\n']))
                f.close()
        p.close()
        p.join()
    if 'DNN' in models:
        dnn = {'split': split, 'FP': FP, 'model': ['DNN']}
        dnn = pd.DataFrame(list(itertools.product(*dnn.values())), columns=dnn.keys())
        for i in range(len(dnn.index.values)):
            start_time = time.time()
            split_type = dnn.iloc[i].split
            FP_type = dnn.iloc[i].FP
            model_type = dnn.iloc[i].model
            a = pair_param(file, data, split_type, model_type, FP_type, difftasks)
            use_time = round(time.time() - start_time, 5)
            local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            if use_time >= 1:
                record_save = file.replace(file.split('/')[-1], 'record_every_model.csv')
                f = open(record_save, 'a+')
                f.write(",".join([file, split_type, model_type, str(use_time), local_time, FP_type, '\n']))
                f.close()
    for i in range(len(dl.index.values)):
        split_type = dl.iloc[i].split
        model_type = dl.iloc[i].model
        start_time = time.time()
        a = pair_param(file, data, split_type, model_type, device, difftasks)
        use_time = int(time.time() - start_time)
        if use_time >= 1:
            local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            record_save = file.replace(file.split('/')[-1], 'record_every_model.csv')
            f = open(record_save, 'a+')
            f.write(",".join([file, split_type, model_type, str(use_time), local_time, device, "\n"]))
            f.close()

    if len(difftasks) > 1:
        if len(set((['KNN', 'SVM', 'RF', 'XGB'])) | set(models)) > 0:
            need = list(set((['KNN', 'SVM', 'RF', 'XGB'])) & set(models))
            file_merge(file, need)


if __name__ == '__main__':
    # args = parse_args()
    # file = args.file
    # split = args.split
    # FP = args.FP
    # model = args.model
    # cpus = args.threads
    # mpl = args.mpl
    # device = args.device
    # main(file, split, FP, model, cpus, mpl, device)
    # running('/data/jianping/bokey/OCAICM/dataset/aurorab/aurorab.csv',out_dir='/data/jianping/bokey/OCAICM/dataset/',split='all',model=['SVM','KNN','XGB','RF','DNN'],FP='all', cpus=8)
    main('/data/jianping/bokey/OCAICM/dataset/aurorab/aurorab.csv',['scaffold'],['ECFP4', 'MACCS'],['gcn'],8,False,'cpu')