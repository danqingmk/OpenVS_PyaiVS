import time
import gc
import os
from dgl.data.utils import Subset
import pandas as pd
from dgl.data.chem import csv_dataset, smiles_to_bigraph, MoleculeCSVDataset
from PyaiVS.gnn_utils import AttentiveFPBondFeaturizer, AttentiveFPAtomFeaturizer, collate_molgraphs, \
    EarlyStopping, set_random_seed, Meter
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
import torch
from PyaiVS.splitdater import split_dataset
from PyaiVS.data_utils import TVT
from dgl.model_zoo.chem import MPNNModel, GCNClassifier, GATClassifier, AttentiveFP
import numpy as np
from sklearn.model_selection import train_test_split
from dgl import backend as F
from hyperopt import fmin, tpe, hp, Trials
epochs = 300
patience = 50
batch_size = 128
start_time = time.time()
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
set_random_seed(seed=42)
torch.set_num_threads(48)

def get_split_index(data, num,split_type='random',random=42):
    # data_tr_x, data_va_x, data_te_x, data_tr_y, data_va_y, data_te_y = split_dataset(X, Y, split_type=split_type,
    #                                                                                  valid_need=True)

    data_x, data_te_x, data_y, data_te_y = data.set2ten(num)
    data_tr_x, data_va_x, data_tr_y, data_va_y = split_dataset(data_x, data_y, split_type=split_type, valid_need=False,
                                                               random_state=random, train_size=(8 / 9))

    tr = pd.DataFrame([data_tr_x, data_tr_y]).T;
    tr['group'] = 'train'
    va = pd.DataFrame([data_va_x, data_va_y]).T;
    va['group'] = 'valid'
    te = pd.DataFrame([data_te_x, data_te_y]).T;
    te['group'] = 'test'
    df = pd.concat([tr, va, te])
    return df[df.group == 'train'].index, df[df.group == 'valid'].index, df[
        df.group == 'test'].index


def run_a_train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()
    train_metric = Meter()  # for each epoch
    for batch_id, batch_data in enumerate(data_loader):

        smiles, bg, labels, masks = batch_data
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        # print(batch_id, smiles[0])
        # transfer the data to device(cpu or cuda)
        labels, masks, atom_feats, bond_feats = labels.to(args['device']), masks.to(args['device']), atom_feats.to(
            args['device']), bond_feats.to(args['device'])

        outputs = model(bg, atom_feats) if args['model'] in ['gcn', 'gat'] else model(bg, atom_feats,
                                                                                               bond_feats)

        loss = (loss_func(outputs, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        outputs.cpu()

        labels.cpu()
        masks.cpu()
        atom_feats.cpu()
        bond_feats.cpu()
        loss.cpu()
        # torch.cuda.empty_cache()

        train_metric.update(outputs, labels, masks)

    if args['metric'] == 'rmse':
        rmse_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(train_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(train_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(train_metric.compute_metric('prc_auc'))  # in case of multi-tasks

        return {'roc_auc': roc_score, 'prc_auc': prc_score}


def run_an_eval_epoch(model, data_loader, args):
    model.eval()
    eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            atom_feats = bg.ndata.pop('h')
            bond_feats = bg.edata.pop('e')

            # transfer the data to device(cpu or cuda)
            labels, masks, atom_feats, bond_feats = labels.to(args['device']), masks.to(args['device']), atom_feats.to(
                args['device']), bond_feats.to(args['device'])
            outputs = model(bg, atom_feats) if args['model'] in ['gcn', 'gat'] else model(bg, atom_feats,
                                                                                                   bond_feats)
            outputs.cpu()
            labels.cpu()
            masks.cpu()
            atom_feats.cpu()
            bond_feats.cpu()
            # loss.cpu()
            torch.cuda.empty_cache()
            eval_metric.update(outputs, labels, masks)
    if args['metric'] == 'rmse':
        rmse_score = np.mean(eval_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(eval_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(eval_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(eval_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(eval_metric.compute_metric('prc_auc'))  # in case of multi-tasks
        se = np.mean(eval_metric.compute_metric('se'), axis=0)
        sp = np.mean(eval_metric.compute_metric('sp'), axis=0)
        acc = np.mean(eval_metric.compute_metric('acc'), axis=0)
        mcc = np.mean(eval_metric.compute_metric('mcc'),axis=0)
        precision = np.mean(eval_metric.compute_metric('precision'),axis=0)
        return {'roc_auc': roc_score, 'prc_auc': prc_score,'se':se,'sp':sp,'acc':acc,'mcc':mcc,'pre':precision}


def get_pos_weight(my_dataset):
    num_pos = F.sum(my_dataset.labels, dim=0)
    num_indices = F.tensor(len(my_dataset.labels))
    return (num_indices - num_pos) / num_pos


def all_one_zeros(series):
    if (len(series.dropna().unique()) == 2):
        flag = False
    else:
        flag = True
    return flag

def best_model_running(seed, opt_res, data, args,file_name,split_type='random', model_name ='gcn',task_type='cla',model_dir=False,my_df=None):

    num_workers =0

    device = 'cpu'

    AtomFeaturizer = AttentiveFPAtomFeaturizer
    BondFeaturizer = AttentiveFPBondFeaturizer
    my_dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(my_df.iloc[:, :], smiles_to_bigraph, AtomFeaturizer,
                                                                    BondFeaturizer, 'Smiles',
                                                                    file_name.replace('.csv', '.bin'))
    if task_type == 'cla':
        pos_weight = get_pos_weight(my_dataset)
    else:
        pos_weight = None
    tasks = args['task']
    tr_indx, val_indx, te_indx = get_split_index(data, seed, split_type='random', random=seed)
    train_loader = DataLoader(Subset(my_dataset, tr_indx), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=num_workers)
    val_loader = DataLoader(Subset(my_dataset, val_indx), batch_size=batch_size, shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=num_workers)
    test_loader = DataLoader(Subset(my_dataset, te_indx), batch_size=batch_size, shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=num_workers)
    # best_model_file = '%s/%s_%s_%s_bst_%s.pth' % (model_dir, args['model'], split_type, args['task'], seed)

    if model_name == 'gcn':
        best_model = GCNClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                   gcn_hidden_feats=opt_res['gcn_hidden_feats'],
                                   n_tasks=len(tasks),
                                   classifier_hidden_feats=opt_res['classifier_hidden_feats'])
        best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s.pth' % (model_dir, args['model'], split_type, 'cla',
                                                             opt_res['l2'], opt_res['lr'],
                                                             opt_res['gcn_hidden_feats'],
                                                             opt_res['classifier_hidden_feats'],seed)

    elif model_name == 'gat':
        best_model = GATClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                   gat_hidden_feats=opt_res['gat_hidden_feats'],
                                   num_heads=opt_res['num_heads'], n_tasks=len(tasks),
                                   classifier_hidden_feats=opt_res['classifier_hidden_feats'])
        best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (model_dir, args['model'], split_type, 'cla',
                                                                opt_res['l2'], opt_res['lr'],
                                                                opt_res['gat_hidden_feats'],
                                                                opt_res['num_heads'],
                                                                opt_res['classifier_hidden_feats'],seed)
    elif model_name == 'attentivefp':
        best_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'),
                                 edge_feat_size=BondFeaturizer.feat_size('e'),
                                 num_layers=opt_res['num_layers'],
                                 num_timesteps=opt_res['num_timesteps'],
                                 graph_feat_size=opt_res['graph_feat_size'], output_size=len(tasks),
                                 dropout=opt_res['dropout'])
        best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s_%s.pth' % (model_dir, args['model'], split_type, 'cla',
                                                                   opt_res['l2'], opt_res['lr'],
                                                                   opt_res['num_layers'],
                                                                   opt_res['num_timesteps'],
                                                                   opt_res['graph_feat_size'],
                                                                   opt_res['dropout'],seed)
    else:
        best_model = MPNNModel(node_input_dim=AtomFeaturizer.feat_size('h'),
                               edge_input_dim=BondFeaturizer.feat_size('e'),
                               output_dim=len(tasks), node_hidden_dim=opt_res['node_hidden_dim'],
                               edge_hidden_dim=opt_res['edge_hidden_dim'],
                               num_layer_set2set=opt_res['num_layer_set2set'])
        best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (model_dir, args['model'], split_type, 'cla',
                                                                opt_res['l2'], opt_res['lr'],
                                                                opt_res['node_hidden_dim'],
                                                                opt_res['edge_hidden_dim'],
                                                                opt_res['num_layer_set2set'],seed)

    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=opt_res['lr'],
                                      weight_decay=opt_res['l2'])
    if task_type == 'reg':
        loss_func = MSELoss(reduction='none')
        stopper = EarlyStopping(mode='lower', patience=patience, filename=best_model_file)
    else:
        loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
        stopper = EarlyStopping(mode='higher', patience=patience, filename=best_model_file)
    # print(best_model_file)
    # best_model.load_state_dict(torch.load(best_model_file, map_location=device)['model_state_dict'])
    # best_model.to(device)

    for j in range(epochs):
        run_a_train_epoch(best_model, train_loader, loss_func, best_optimizer, args)
        train_scores = run_an_eval_epoch(best_model, train_loader, args)
        val_scores = run_an_eval_epoch(best_model, val_loader, args)
        early_stop = stopper.step(val_scores[args['metric']], best_model)
        # print(j,val_scores)
        if early_stop:
            # print(j,val_scores)
            break
    stopper.load_checkpoint(best_model)
    tr_scores = run_an_eval_epoch(best_model, train_loader, args)
    val_scores = run_an_eval_epoch(best_model, val_loader, args)
    te_scores = run_an_eval_epoch(best_model, test_loader, args)
    result_one = pd.concat([pd.DataFrame([tr_scores],index=['tr']),pd.DataFrame([val_scores],index=['va']),pd.DataFrame([te_scores],index=['te'])])
    result_one['type'] = result_one.index
    result_one['split'] = split_type
    result_one['model'] = model_name
    result_one['seed'] = seed
    result_one.columns = ['auc_roc','auc_prc', 'se', 'sp', 'acc', 'mcc', 'precision', 'type', 'split','model','seed']


    return result_one

def tvt_dl(X,Y,split_type='random',model_name='gcn',task_type='cla',file_name=None,model_dir=None,device ='cpu',difftasks='activity'):

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    device = device if device =='cpu' else 'cuda:0'
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # if device == 'cuda':
    #    torch.cuda.set_device(eval(gpu_id))  # gpu device id
    file_name = file_name.replace('.csv', '_pro.csv')
    my_df = pd.read_csv(file_name)


    AtomFeaturizer = AttentiveFPAtomFeaturizer
    BondFeaturizer = AttentiveFPBondFeaturizer

    opt_iters =50
    num_workers = 0
    args = {'device': device, 'task': difftasks,
            'metric': 'roc_auc' if task_type == 'cla' else 'rmse', 'model': model_name}
    tasks = args['task']

    Hspace = {'gcn': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                          lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                          gcn_hidden_feats=hp.choice('gcn_hidden_feats',
                                                     [[128, 128], [256, 256], [128, 64], [256, 128]]),
                          classifier_hidden_feats=hp.choice('classifier_hidden_feats', [128, 64, 256])),
              'mpnn': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                           lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                           node_hidden_dim=hp.choice('node_hidden_dim', [64, 32, 16]),
                           edge_hidden_dim=hp.choice('edge_hidden_dim', [64, 32, 16]),
                           num_layer_set2set=hp.choice('num_layer_set2set', [2, 3, 4])),
              'gat': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                          lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                          gat_hidden_feats=hp.choice('gat_hidden_feats',
                                                     [[128, 128], [256, 256], [128, 64], [256, 128]]),
                          num_heads=hp.choice('num_heads', [[2, 2], [3, 3], [4, 4], [4, 3], [3, 2]]),
                          classifier_hidden_feats=hp.choice('classifier_hidden_feats', [128, 64, 256])),
              'attentivefp': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                                  lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                                  num_layers=hp.choice('num_layers', [2, 3, 4, 5, 6]),
                                  num_timesteps=hp.choice('num_timesteps', [1, 2, 3, 4, 5]),
                                  dropout=hp.choice('dropout', [ 0.1, 0.3, 0.5]),
                                  graph_feat_size=hp.choice('graph_feat_size', [50, 100, 200, 300]))}
    hyper_space = Hspace[args['model']]


    my_dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(my_df.iloc[:, :], smiles_to_bigraph, AtomFeaturizer,
                                                                    BondFeaturizer, 'Smiles',
                                                                    file_name.replace('.csv', '.bin'))
    if task_type == 'cla':
        pos_weight = get_pos_weight(my_dataset)
    else:
        pos_weight = None

    data = TVT(X, Y)
    tr_indx, val_indx, te_indx = get_split_index(data,0,split_type=split_type)
    train_loader = DataLoader(Subset(my_dataset, tr_indx), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=num_workers)
    val_loader = DataLoader(Subset(my_dataset, val_indx), batch_size=batch_size, shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=num_workers)
    test_loader = DataLoader(Subset(my_dataset, te_indx), batch_size=batch_size, shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=num_workers)


    def hyper_opt(hyper_paras):
        # get the model instance
        if model_name == 'gcn':
            my_model = GCNClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                     gcn_hidden_feats=hyper_paras['gcn_hidden_feats'],
                                     n_tasks=len(tasks), classifier_hidden_feats=hyper_paras['classifier_hidden_feats'])
            model_file_name = '%s/%s_%s_%s_%s_%.6f_%s_%s.pth' % (model_dir, args['model'], split_type, 'cla',
                                                                 hyper_paras['l2'], hyper_paras['lr'],
                                                                 hyper_paras['gcn_hidden_feats'],
                                                                 hyper_paras['classifier_hidden_feats'])

        elif model_name == 'mpnn':
            my_model = MPNNModel(node_input_dim=AtomFeaturizer.feat_size('h'), edge_input_dim=BondFeaturizer.feat_size('e'),
                                 output_dim=len(tasks), node_hidden_dim=hyper_paras['node_hidden_dim'],
                                 edge_hidden_dim=hyper_paras['edge_hidden_dim'],
                                 num_layer_set2set=hyper_paras['num_layer_set2set'])
            model_file_name = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s.pth' % (model_dir, args['model'], split_type, 'cla',
                                                                    hyper_paras['l2'], hyper_paras['lr'],
                                                                    hyper_paras['node_hidden_dim'],
                                                                    hyper_paras['edge_hidden_dim'],
                                                                    hyper_paras['num_layer_set2set'])
        elif model_name == 'attentivefp':
            my_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'),
                                   edge_feat_size=BondFeaturizer.feat_size('e'),
                                   num_layers=hyper_paras['num_layers'], num_timesteps=hyper_paras['num_timesteps'],
                                   graph_feat_size=hyper_paras['graph_feat_size'], output_size=len(tasks),
                                   dropout=hyper_paras['dropout'])
            model_file_name = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (
            model_dir, args['model'], split_type, 'cla',
            hyper_paras['l2'], hyper_paras['lr'],
            hyper_paras['num_layers'],
            hyper_paras['num_timesteps'],
            hyper_paras['graph_feat_size'],
            hyper_paras['dropout'])

        else:
            my_model = GATClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                     gat_hidden_feats=hyper_paras['gat_hidden_feats'],
                                     num_heads=hyper_paras['num_heads'], n_tasks=len(tasks),
                                     classifier_hidden_feats=hyper_paras['classifier_hidden_feats'])
            model_file_name = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s.pth' % (model_dir, args['model'], split_type, 'cla',
                                                                    hyper_paras['l2'], hyper_paras['lr'],
                                                                    hyper_paras['gat_hidden_feats'],
                                                                    hyper_paras['num_heads'],
                                                                    hyper_paras['classifier_hidden_feats'])

        optimizer = torch.optim.Adam(my_model.parameters(), lr=hyper_paras['lr'], weight_decay=hyper_paras['l2'])

        if task_type == 'reg':
            loss_func = MSELoss(reduction='none')
            stopper = EarlyStopping(mode='lower', patience=patience,filename=model_file_name)
        else:
            loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
            stopper = EarlyStopping(mode='higher', patience=patience,filename=model_file_name)
        my_model.to(device)

        for j in range(epochs):
            # training
            run_a_train_epoch(my_model, train_loader, loss_func, optimizer, args)

            # early stopping
            val_scores = run_an_eval_epoch(my_model, val_loader, args)
            early_stop = stopper.step(val_scores[args['metric']], my_model)

            if early_stop:
                break
        stopper.load_checkpoint(my_model)
        # tr_scores = run_an_eval_epoch(my_model, train_loader, args)
        val_scores = run_an_eval_epoch(my_model, val_loader, args)
        # te_scores = run_an_eval_epoch(my_model, test_loader, args)

        feedback = val_scores[args['metric']] if task_type == 'reg' else (1 - val_scores[args['metric']])
        my_model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        return feedback


    # start hyper-parameters optimization

    trials = Trials()
    opt_res = fmin(hyper_opt, hyper_space, algo=tpe.suggest, max_evals=opt_iters, trials=trials)

   # construct the model based on the optimal hyper-parameters
    l2_ls = [0, 10 ** -8, 10 ** -6, 10 ** -4]
    lr_ls = [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]
    hidden_feats_ls = [(128, 128), (256, 256), (128, 64), (256, 128)]
    classifier_hidden_feats_ls = [128, 64, 256]
    node_hidden_dim_ls = [64, 32, 16]
    edge_hidden_dim_ls = [64, 32, 16]
    num_layer_set2set_ls = [2, 3, 4]
    num_heads_ls = [(2, 2), (3, 3), (4, 4), (4, 3), (3, 2)]
    num_layers_ls = [2, 3, 4, 5, 6]
    num_timesteps_ls = [1, 2, 3, 4, 5]
    graph_feat_size_ls = [50, 100, 200, 300]
    dropout_ls = [ 0.1, 0.3, 0.5]
    if model_name == 'gcn':
        param = {'l2':l2_ls[opt_res['l2']], 'lr':lr_ls[opt_res['lr']],
             'gcn_hidden_feats':hidden_feats_ls[opt_res['gcn_hidden_feats']],
             'classifier_hidden_feats':classifier_hidden_feats_ls[opt_res['classifier_hidden_feats']]}
    elif model_name == 'mpnn':
        param = {'l2': l2_ls[opt_res['l2']], 'lr': lr_ls[opt_res['lr']],
                 'node_hidden_dim': node_hidden_dim_ls[opt_res['node_hidden_dim']],
                 'edge_hidden_dim': edge_hidden_dim_ls[opt_res['edge_hidden_dim']],
                 'num_layer_set2set': num_layer_set2set_ls[opt_res['num_layer_set2set']]}
    elif model_name == 'attentivefp':
        param = {'l2': l2_ls[opt_res['l2']], 'lr': lr_ls[opt_res['lr']],
                 'num_layers': num_layers_ls[opt_res['num_layers']],
                 'num_timesteps': num_timesteps_ls[opt_res['num_timesteps']],
                 'graph_feat_size': graph_feat_size_ls[opt_res['graph_feat_size']],
                 'dropout':dropout_ls[opt_res['dropout']]}

    else:
        param = {'l2': l2_ls[opt_res['l2']], 'lr': lr_ls[opt_res['lr']],
                 'gat_hidden_feats': hidden_feats_ls[opt_res['gat_hidden_feats']],
                 'num_heads': num_heads_ls[opt_res['num_heads']],
                 'classifier_hidden_feats': classifier_hidden_feats_ls[opt_res['classifier_hidden_feats']]}

    para_file = str(model_dir).replace('model_save', 'param_save') + '/%s_%s_%s' % (
        split_type, task_type,  '{}.param'.format(model_name))
    if not os.path.exists(str(model_dir).replace('model_save', 'param_save')):
        os.makedirs(str(model_dir).replace('model_save', 'param_save'))
    print(os.path.exists(str(model_dir).replace('model_save', 'param_save')))
    f = open(para_file, 'w')
    f.write('%s' % param)
    f.close()


def para_dl(X,Y,opt_res=None,split_type='random',model_name='gcn',task_type='cla',file_name=None,model_dir=None,device ='cpu',difftasks=None):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = device if device =='cpu' else 'cuda:0'
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # if device == 'cuda':
    #    torch.cuda.set_device(eval(gpu_id))  # gpu device id
    file_name = file_name.replace('.csv', '_pro.csv')
    opt_res = eval(open(str(model_dir).replace('model_save', 'param_save') + '/%s_%s_%s' % (
        split_type, task_type,  '{}.param'.format(model_name)), 'r').readline().strip()) if opt_res == None else opt_res

    my_df = pd.read_csv(file_name)
    AtomFeaturizer = AttentiveFPAtomFeaturizer
    BondFeaturizer = AttentiveFPBondFeaturizer

    repetitions = 9
    num_workers = 0

    args = {'device': device, 'task': difftasks,
            'metric': 'roc_auc' if task_type == 'cla' else 'rmse', 'model': model_name}
    tasks = args['task']
    my_dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(my_df.iloc[:, :], smiles_to_bigraph, AtomFeaturizer,
                                                                    BondFeaturizer, 'Smiles',
                                                                    file_name.replace('.csv', '.bin'))
    if task_type == 'cla':
        pos_weight = get_pos_weight(my_dataset)
    else:
        pos_weight = None

    data = TVT(X, Y)
    tr_indx, val_indx, te_indx = get_split_index(data,0,split_type=split_type,random=42)
    train_loader = DataLoader(Subset(my_dataset, tr_indx), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=num_workers)
    val_loader = DataLoader(Subset(my_dataset, val_indx), batch_size=batch_size, shuffle=False,
                            collate_fn=collate_molgraphs, num_workers=num_workers)
    test_loader = DataLoader(Subset(my_dataset, te_indx), batch_size=batch_size, shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=num_workers)

    if model_name == 'gcn':
        best_model = GCNClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                   gcn_hidden_feats=opt_res['gcn_hidden_feats'],
                                   n_tasks=len(tasks),
                                   classifier_hidden_feats=opt_res['classifier_hidden_feats'])
        best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s.pth' % (model_dir,args['model'], split_type,'cla',
                                                                         opt_res['l2'], opt_res['lr'],
                                                                         opt_res['gcn_hidden_feats'],
                                                                         opt_res['classifier_hidden_feats'])

        param = {'l2':opt_res['l2'], 'lr':opt_res['lr'],
             'gcn_hidden_feat':opt_res['gcn_hidden_feats'],
             'classifier_hidden_feats':opt_res['classifier_hidden_feats']}
    elif model_name == 'mpnn':
        best_model = MPNNModel(node_input_dim=AtomFeaturizer.feat_size('h'), edge_input_dim=BondFeaturizer.feat_size('e'),
                               output_dim=len(tasks), node_hidden_dim=opt_res['node_hidden_dim'],
                               edge_hidden_dim=opt_res['edge_hidden_dim'],
                               num_layer_set2set=opt_res['num_layer_set2set'])
        best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s.pth' % (model_dir,args['model'], split_type, 'cla',
                                                                        opt_res['l2'], opt_res['lr'],
                                                                        opt_res['node_hidden_dim'],
                                                                        opt_res['edge_hidden_dim'],
                                                                        opt_res['num_layer_set2set'])
        param = {'l2': opt_res['l2'], 'lr': opt_res['lr'],
                 'node_hidden_dim': opt_res['node_hidden_dim'],
                 'edge_hidden_dim': opt_res['edge_hidden_dim'],
                 'num_layer_set2set': opt_res['num_layer_set2set']}
    elif model_name == 'attentivefp':
        best_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'), edge_feat_size=BondFeaturizer.feat_size('e'),
                               num_layers=opt_res['num_layers'],
                               num_timesteps=opt_res['num_timesteps'],
                               graph_feat_size=opt_res['graph_feat_size'], output_size=len(tasks),
                               dropout=opt_res['dropout'])
        best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (model_dir,args['model'], split_type, 'cla',
                                                                           opt_res['l2'], opt_res['lr'],
                                                                                   opt_res['num_layers'],
                                                                                   opt_res['num_timesteps'],
                                                                                   opt_res['graph_feat_size'],
                                                                                   opt_res['dropout'])
        param = {'l2': opt_res['l2'], 'lr': opt_res['lr'],
                 'num_layers': opt_res['num_layers'],
                 'num_timesteps': opt_res['num_timesteps'],
                 'graph_feat_size': opt_res['graph_feat_size'],
                 'dropout':opt_res['dropout']}

    else:
        best_model = GATClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                   gat_hidden_feats=opt_res['gat_hidden_feats'],
                                   num_heads=opt_res['num_heads'], n_tasks=len(tasks),
                                   classifier_hidden_feats=opt_res['classifier_hidden_feats'])
        best_model_file = '%s/%s_%s_%s_%s_%.6f_%s_%s_%s.pth' % (model_dir,args['model'], split_type, 'cla',
                                                                        opt_res['l2'], opt_res['lr'],
                                                                                opt_res['gat_hidden_feats'],
                                                                                opt_res['num_heads'],
                                                                                opt_res['classifier_hidden_feats'])
        param = {'l2': opt_res['l2'], 'lr': opt_res['lr'],
                 'gat_hidden_feats': opt_res['gat_hidden_feats'],
                 'num_heads': opt_res['num_heads'],
                 'classifier_hidden_feats': opt_res['classifier_hidden_feats']}

    # best_model.load_state_dict(torch.load(best_model_file, map_location=device)['model_state_dict'])
    # best_model.to(device)

    optimizer = torch.optim.Adadelta(best_model.parameters(), lr=opt_res['lr'], weight_decay=opt_res['l2'])

    if task_type == 'reg':
        loss_func = MSELoss(reduction='none')
        stopper = EarlyStopping(mode='lower', patience=patience, filename=best_model_file)
    else:
        loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))  #

        stopper = EarlyStopping(mode='higher', patience=patience, filename=best_model_file)
    best_model.to(device)
    for j in range(epochs):
        run_a_train_epoch(best_model, train_loader, loss_func, optimizer, args)
        # early stopping
        val_scores = run_an_eval_epoch(best_model, val_loader, args)
        early_stop = stopper.step(val_scores[args['metric']], best_model)

        if early_stop:
            break
    stopper.load_checkpoint(best_model)



    tr_scores = run_an_eval_epoch(best_model, train_loader, args)
    val_scores = run_an_eval_epoch(best_model, val_loader, args)
    te_scores = run_an_eval_epoch(best_model, test_loader, args)

    record = {'auc_roc':[tr_scores['roc_auc'],val_scores['roc_auc'],te_scores['roc_auc']],
              'auc_prc':[tr_scores['prc_auc'],val_scores['prc_auc'],te_scores['prc_auc']],
              'se':[tr_scores['se'],val_scores['se'],te_scores['se']],
              'sp':[tr_scores['sp'],val_scores['sp'],te_scores['sp']],
              'acc':[tr_scores['acc'],val_scores['acc'],te_scores['acc']],
              'mcc':[tr_scores['mcc'],val_scores['mcc'],te_scores['mcc']],
              'precision':[tr_scores['pre'],val_scores['pre'],te_scores['pre']]}
    param = {k:[v,v,v] for k,v in param.items()}
    record = {k:v for d in [param,record] for k,v in d.items()}
    best_res = pd.DataFrame(record,index=['tr','va','te'])
    best_res['type'] = best_res.index
    best_res['split'] = split_type
    best_res['model'] = model_name
    best_res['seed'] = 0

    para_res = best_res[[ 'auc_roc',
       'auc_prc', 'se', 'sp', 'acc', 'mcc', 'precision', 'type', 'split',
       'model','seed']]
    for seed in range(1,repetitions+1):
        res_best = best_model_running(seed, opt_res, data, args, file_name, split_type=split_type, model_name=model_name, task_type=task_type,
                       model_dir=model_dir, my_df=my_df)
        para_res = pd.concat([para_res,res_best],ignore_index=True)
    result_dir = model_dir.replace('model_save', 'result_save')
    para_name, best_name = os.path.join(result_dir,
                                        '_'.join([split_type, model_name,'para.csv'])), os.path.join(
        result_dir, '_'.join([split_type, model_name,'best.csv']))
    para_res.to_csv(best_name, index=False)
    best_res.to_csv(para_name, index=False)
    print(para_res.groupby(['split', 'type'])['acc', 'mcc', 'auc_prc', 'auc_roc'].mean(),best_res)
    return  para_res.groupby(['split', 'type'])['acc', 'mcc', 'auc_prc', 'auc_roc'].mean(),best_res
    # print(para_res.groupby(['split', 'type'])['acc', 'mcc', 'auc_prc', 'auc_roc'].mean())

    # for seed in range(1, repetitions + 1):
    #
    #     tr_indx, val_indx, te_indx = get_split_index(data,seed,split_type='random',random=seed)
    #     train_loader = DataLoader(Subset(my_dataset, tr_indx), batch_size=batch_size, shuffle=True,
    #                               collate_fn=collate_molgraphs, num_workers=num_workers)
    #     val_loader = DataLoader(Subset(my_dataset, val_indx), batch_size=batch_size, shuffle=False,
    #                             collate_fn=collate_molgraphs, num_workers=num_workers)
    #     test_loader = DataLoader(Subset(my_dataset, te_indx), batch_size=batch_size, shuffle=False,
    #                              collate_fn=collate_molgraphs, num_workers=num_workers)
    #     best_model_file = '%s/%s_%s_%s_bst_%s.pth' % (model_dir,args['model'], split_type, args['task'], seed)
    #
    #     if model_name == 'gcn':
    #         best_model = GCNClassifier(in_feats=AtomFeaturizer.feat_size('h'),
    #                                    gcn_hidden_feats=opt_res['gcn_hidden_feats'],
    #                                    n_tasks=len(tasks),
    #                                    classifier_hidden_feats=opt_res['classifier_hidden_feats'])
    #
    #     elif model_name == 'gat':
    #         best_model = GATClassifier(in_feats=AtomFeaturizer.feat_size('h'),
    #                                    gat_hidden_feats=opt_res['gat_hidden_feats'],
    #                                    num_heads=opt_res['num_heads'], n_tasks=len(tasks),
    #                                    classifier_hidden_feats=opt_res['classifier_hidden_feats'])
    #     elif model_name == 'attentivefp':
    #         best_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'),
    #                                  edge_feat_size=BondFeaturizer.feat_size('e'),
    #                                  num_layers=opt_res['num_layers'],
    #                                  num_timesteps=opt_res['num_timesteps'],
    #                                  graph_feat_size=opt_res['graph_feat_size'], output_size=len(tasks),
    #                                  dropout=opt_res['dropout'])
    #     else:
    #         best_model = MPNNModel(node_input_dim=AtomFeaturizer.feat_size('h'),
    #                                edge_input_dim=BondFeaturizer.feat_size('e'),
    #                                output_dim=len(tasks), node_hidden_dim=opt_res['node_hidden_dim'],
    #                                edge_hidden_dim=opt_res['edge_hidden_dim'],
    #                                num_layer_set2set=opt_res['num_layer_set2set'])
    #
    #     best_optimizer = torch.optim.Adam(best_model.parameters(), lr=opt_res['lr'],
    #                                       weight_decay=opt_res['l2'])
    #     if task_type == 'reg':
    #         loss_func = MSELoss(reduction='none')
    #         stopper = EarlyStopping(mode='lower', patience=patience, filename=best_model_file)
    #     else:
    #         loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
    #         stopper = EarlyStopping(mode='higher', patience=patience,filename=best_model_file)
    #     best_model.to(device)
    #
    #     for j in range(epochs):
    #         run_a_train_epoch(best_model, train_loader, loss_func, best_optimizer, args)
    #         train_scores = run_an_eval_epoch(best_model, train_loader, args)
    #         val_scores = run_an_eval_epoch(best_model, val_loader, args)
    #         early_stop = stopper.step(val_scores[args['metric']], best_model)
    #         if early_stop:
    #             break
    #     stopper.load_checkpoint(best_model)
    #     tr_scores = run_an_eval_epoch(best_model, train_loader, args)
    #     val_scores = run_an_eval_epoch(best_model, val_loader, args)
    #     te_scores = run_an_eval_epoch(best_model, test_loader, args)
    #
    #     tr_res.append(tr_scores);
    #     val_res.append(val_scores);
    #     te_res.append(te_scores)
    # if task_type == 'reg':
    #     cols = ['rmse', 'mae', 'r2']
    #     pd1 = best_res[['FP_type', 'split', 'type',
    #                     'num_of_compounds', 'rmse', 'r2', 'mae']]
    #     pd1['seed'] = 0
    # else:
    #     cols = ['auc_roc', 'auc_prc','se','sp','acc','mcc','precision']
    #     pd1 = best_res[[ 'model','split', 'type','auc_roc', 'auc_prc','se','sp','acc','mcc','precision']]
    #     pd1['seed'] = 0
    # tr = [list(item.values()) for item in tr_res]
    # val = [list(item.values()) for item in val_res]
    # te = [list(item.values()) for item in te_res]
    # tr_pd = pd.DataFrame(tr, columns=cols)
    # tr_pd['seed'] = [seed for seed in range(1, repetitions + 1)]
    # tr_pd['type'] = 'tr'
    # val_pd = pd.DataFrame(val, columns=cols)
    # val_pd['seed'] = [seed for seed in range(1, repetitions + 1)]
    # val_pd['type'] = 'va'
    # te_pd = pd.DataFrame(te, columns=cols)
    # te_pd['seed'] = [seed for seed in range(1, repetitions + 1)]
    # te_pd['type'] = 'te'
    # sta_pd = pd.concat([tr_pd, val_pd, te_pd], ignore_index=True)
    # sta_pd['model'] = args['model']
    # sta_pd['split'] = split_type
    # sta_pd = pd.concat([pd1, sta_pd], ignore_index=True)
    # return best_res,sta_pd
# df = pd.read_csv('/data/jianping/bokey/OCAICM/dataset/aurorab/aurorab.csv')
# a,b = para_dl(df['Smiles'],df['activity'],opt_res=None,split_type='scaffold',model_name='gcn',task_type='cla'
#               ,file_name='/data/jianping/bokey/OCAICM/dataset/aurorab/aurorab.csv',
#               model_dir='/data/jianping/bokey/OCAICM/dataset/aurorab/model_save/gcn',device ='cpu',difftasks=['activity'])
#
# print(a,b[['acc','auc_prc','mcc','auc_roc']])