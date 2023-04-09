import os
import pandas as pd
import numpy as np
from dgl.data.chem import csv_dataset, smiles_to_bigraph, MoleculeCSVDataset
from torch.utils.data import DataLoader
from dgl.model_zoo.chem import MPNNModel, GCNClassifier, GATClassifier, AttentiveFP
from PyaiVS.dnn_torch_utils import Meter, MyDataset, EarlyStopping, MyDNN, collate_fn, set_random_seed
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import gc
import time
start_time = time.time()
import warnings
from sklearn import preprocessing
from PyaiVS.splitdater import split_dataset
from PyaiVS.feature_create import create_des
from PyaiVS.data_utils import TVT
import os
import time
s =time.time()
batch_size = 1
def complex_ele(models,splits,FPs):
    info =[]
    for model in models:
        for split in splits:
            for FP in FPs:
                info.append((model,split,FP))
    return info
import torch
torch.set_num_threads(48)
# def run_an_eval_epoch(model, data_loader, args):
#     f = open(args['output'],'w+')
#
#     f.write('cano_smiles,pred_prop\n')
#     # print(args['output'])
#     model.eval()
#     # eval_metric = Meter()
#     smile_list = {}
#     count  = 0
#     with torch.no_grad():
#         for batch_id, batch_data in enumerate(data_loader):
#             eval_metric = Meter()
#             smiles, bg, labels, masks = batch_data
#             smile_list[count]=smiles
#             atom_feats = bg.ndata.pop('h')
#             bond_feats = bg.edata.pop('e')
#             # transfer the data to device(cpu or cuda)
#             outputs = model(bg, atom_feats) if args['model'] in ['gcn', 'gat'] else model(bg, atom_feats,bond_feats)
#
#
#             # smile_list.append(smiles)
#             eval_metric.update(outputs, labels, torch.tensor([count]))
#             roc_score = eval_metric.compute_metric('pred')
#
#             if roc_score.tolist()[0][0] >= args['prop']:
#                 f.write('{},{}\n'.format(smiles[0],round(roc_score.tolist()[0][0],2)))
#                 #
#             count += 1
#             if count%100000 ==0:
#                 print(count)
#             torch.cuda.empty_cache()
#         f.close()

def run_an_eval_epoch(model, data_loader, args):
    f = open(args['output'], 'w+')

    f.write('cano_smiles,pred_prop\n')
    # print(args['output'])
    model.eval()
    # eval_metric = Meter()
    smile_list = {}
    count = 0
    model.eval()


    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            eval_metric = Meter()
            Xs, Ys, masks = batch_data
            # transfer the data to device(cpu or cuda)
            outputs = model(Xs)
            # print(outputs)
            my_df= args['data']
            smile_list[count] = my_df.loc[count,args['smiles_col']]
            # print(my_df.loc[count,args['smiles_col']])
            outputs.cpu()
            Ys.cpu()
            masks.cpu()
#            torch.cuda.empty_cache()
            eval_metric.update(outputs, Ys, torch.tensor([count]))

            roc_score = eval_metric.compute_metric('pred')
            if roc_score.tolist()[0][0] >= args['prop']:
                f.write('{},{}\n'.format(my_df.loc[count,args['smiles_col']], round(roc_score.tolist()[0][0], 2)))
                #
            count += 1
            if count % 100000 == 0:
                print(count)
            torch.cuda.empty_cache()
    f.close()

def screen(args,file=None):

    if os.path.exists(args['output']):
        print(args['output'],'has done')
    else:
        my_df = pd.read_csv(file, sep=args['sep'], usecols=[0],  engine='python')
        my_df.columns = [args['smiles_col']]
        my_df['index'] = my_df.index
        my_df = my_df[[args['smiles_col'],'index']]
        opt_res = eval(open(args['param_file'], 'r').readline().strip())

        data_tr_x, data_tr_y = create_des(my_df[args['smiles_col']], my_df['index'], FP_type=args['des'], model_dir=model_dir)
        data_tr_y = data_tr_y.values.reshape(-1, 1)
        inputs = data_tr_x.shape[1]
        dataset_loader = DataLoader(MyDataset(data_tr_x, data_tr_y), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        best_model = MyDNN(inputs=inputs, hideen_units=opt_res['hidden_units'], outputs=1,
                           dp_ratio=opt_res['dropout'], reg=False)



        args['data']=my_df
        print(args['model_file'])
        best_model.load_state_dict(
            torch.load(args['model_file'], map_location='cpu')['model_state_dict'])
        run_an_eval_epoch(best_model, dataset_loader, args)
import sys
def find_path(truple_ele):
    model, split, FP = truple_ele
    if model in ['SVM', 'KNN', 'RF', 'XGB']:
        model_file = '_'.join([split, 'cla', FP, model, 'bestModel.pkl'])
        model_path = os.path.join(os.path.join(model_dir, model), model_file)
        model_paths = [model_path, os.path.join(os.path.join(model_dir, model),
                                                '_'.join([split, 'cla', FP, 'SVM', 'bestModel.pkl']))]
        model_path = ''
        for path in model_paths:
            if os.path.exists(path):
                model_path += path
                break
        return model_path
    elif model == 'DNN':
        param_dir = model_dir.replace('model_save', 'param_save')
        opt_res = eval(
            open(os.path.join(param_dir, model, '{}_cla_{}_{}.param'.format(split, FP, model)), 'r').readline().strip())
        model_path = '%s/%s/%s_%s_%s_%s_%.4f_%d_%d_%d.4f_early_stop_%d.pth' % (
            model_dir, model, 'cla', FP, split, 'dataset', opt_res['dropout'],
            opt_res['hidden_units'][0],
            opt_res['hidden_units'][1],
            opt_res['hidden_units'][2],
            opt_res['l2'])
        return model_path
    else:
        param_dir = model_dir.replace('model_save', 'param_save')
        param_file = os.path.join(param_dir, model, '{}_cla_{}.param'.format(split, model))
        param = open(param_file,'r').readline().strip()

        if model == 'gcn':
            opt_res = eval(param)
            model_path = '%s/%s/%s_%s_%s_%s_%.6f_%s_%s.pth' % (model_dir, model, model, split, '[\'Activity\']',
                                                               opt_res['l2'], opt_res['lr'],
                                                               opt_res['gcn_hidden_feats'],
                                                               opt_res['classifier_hidden_feats'])

        elif model == 'gat':
            opt_res = eval(param)
            model_path = '%s/%s/%s_%s_%s_%s_%.6f_%s_%s_%s.pth' % (model_dir, model, model, split, '[\'Activity\']',
                                                                  opt_res['l2'], opt_res['lr'],
                                                                  opt_res['gat_hidden_feats'],
                                                                  opt_res['num_heads'],
                                                                  opt_res['classifier_hidden_feats'])

        else:
            opt_res = eval(param)
            model_path = '%s/%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (model_dir, model, model, split, '[\'Activity\']',
                                                                     opt_res['l2'], opt_res['lr'],
                                                                     opt_res['num_layers'],
                                                                     opt_res['num_timesteps'],
                                                                     opt_res['graph_feat_size'],
                                                                     opt_res['dropout'])

        return model_path ,param_file


# if not os.path.exists(output):
# #     os.makedirs(output)
# # file = dir
# # if os.path.isdir(file):
# #     p = mp.Pool(processes=3)
# #     for file_content in os.listdir(file):
# #         print(time.time())
# #
# #         file_path = os.path.join(file,file_content)
# #         param = {'file':file_path,  'prop':0.8, 'smiles_col':0,'output':output}
# #         get = p.apply_async(screen,kwds = param)
# #     p.close()
# #     p.join()
# # elif os.path.isfile(file):
# #     screen(file=file,prop=0.8,smiles_col='smiles',output =output)
# # else:
# #     pass
# # print(time.time()-s,'cpu=','30')
#
# # def mol_check(smiles):
# #     try:
# #         mol = AllChem.MolFromSmiles(smiles)
# #         rdmolfiles.CanonicalRankAtoms(mol)
# #         return 1
# #     except:
# #         return 0
# # def datacheck(file):
# #
# #     my_df = pd.read_csv(file, sep=';', usecols=[i for i in range(5)], engine='python')
# #     my_df.columns = open(file, 'r').readline().split(';')[:5]
# #     # print('pass')
# #     # my_df.columns = [smiles_col]
# #     # my_df = pd.read_csv(file,sep=';')
# #     my_df['index'] = my_df.index
# #     my_df = my_df[['smiles', 'index']]
# #     my_df['check'] = my_df['smiles'].apply(mol_check)
# #     my_df =my_df[my_df['check']==1]
# #     output =file.replace('.csv','_graph.csv')
# #     my_df.to_csv(output,index=False)
# # datacheck(dir)
# #
# #
# # p = mp.Pool(processes=6)
# # for file in file_list:
# #     file =file.replace('.csv','_graph.csv')
# #     screen(file=file,prop=0.5,smiles_col='smiles')
# # p.close()
# # p.join()
dataset = 'chembl31'
model_dir = '/data/jianping/bokey/OCAICM/dataset/{0}/{0}/model_save'.format(dataset)
dir = '/data/jianping/enamine/D001-1_graph.csv'
info =pd.read_csv('/data/jianping/bokey/OCAICM/dataset/{0}/{0}_record_max.csv'.format(dataset),sep=';')
models = ['DNN']
splits = ['cluster','scaffold','random']#
FPs = ['ECFP4','MACCS']#
comp = complex_ele(models,splits,FPs)
args = {'prop':0.8,'smiles_col':'smiles','sep':','}
for element in comp:
    output = '/data/jianping/bokey/OCAICM/dataset/{}/'.format(dataset)
    data = info[(info['model']==element[0])&(info['split'] ==element[1])&(info['des'] ==element[2])]#
    data=data.reset_index()
    print(data)
    args['param_file'] = data.loc[0,'param_file']
    args['model_file'] = data.loc[0,'model_file']
    print(data.loc[0,'model_file'])
    args['model'] = element[0]
    args['des'] = element[2]
    output = os.path.join(output,'_'.join(list(element)))+'_screen_{}.csv'.format(args['prop'])
    print(output)
    args['output'] = output
    screen(args,file=dir)
    # for file in os.listdir(dir):
    #     if '.csv' in file:
    #         outputs = os.path.join(output,file)
    #         outputs = outputs if ' ' not in outputs else outputs.replace(' ','')
    #         args['output'] = outputs
    #         print(file)
    #         screen(args,file=os.path.join(dir,file))
