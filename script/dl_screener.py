import torch
import os
import numpy as np
import pandas as pd
from PyaiVS.dnn_torch_utils import Meter, MyDataset, EarlyStopping, MyDNN, collate_fn, set_random_seed
from dgl.data.chem import csv_dataset, smiles_to_bigraph, MoleculeCSVDataset
from dgl.model_zoo.chem import MPNNModel, GCNClassifier, GATClassifier, AttentiveFP
from PyaiVS.gnn_utils import AttentiveFPBondFeaturizer, AttentiveFPAtomFeaturizer, collate_molgraphs
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials, partial
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import gc
import time
start_time = time.time()
import warnings
from sklearn import preprocessing
from PyaiVS.splitdater import split_dataset
from PyaiVS.feature_create import create_des
import torch
torch.set_num_threads(5)

def run_an_eval_epoch(model, data_loader, args):
    # f = open(args['output'],'w+')
    # f.write('cano_smiles,pred_prop\n')
    top_k_file=args['output'].replace('.csv', '_top_{}.csv'.format(args['top_k']))
    data=args['data']
    del data["label"]
    out=[]
    count = 0
    model.eval()
    # eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            eval_metric = Meter()
            # Xs, Ys, masks = batch_data
            # Xs, Ys, masks = Xs.to(args['device']), Ys.to(args['device']), masks.to(args['device'])
            # outputs = model(Xs)
            
            smiles, bg, Ys, masks = batch_data
            atom_feats = bg.ndata.pop('h')
            bond_feats = bg.edata.pop('e')
            
            Ys, masks, atom_feats, bond_feats = Ys.to(args['device']), masks.to(args['device']), atom_feats.to(
                args['device']), bond_feats.to(args['device'])
            outputs = model(bg, atom_feats) if args['model_name'] in ['gcn', 'gat'] else model(bg, atom_feats,
                                                                                                   bond_feats)
            outputs.cpu()
            Ys.cpu()
            masks.cpu()
#            torch.cuda.empty_cache()
            
            if args['reg']:
                roc_score = outputs
                out.append(float(roc_score[0].cpu().numpy()))
            else:
                eval_metric.update(outputs, Ys, torch.tensor([count]))
                roc_score = eval_metric.compute_metric('pred')
                smiles = args['data'][args['smiles_col']].tolist()[int(Ys[0])]
                out.append([float(roc_score[0].numpy()), int(roc_score[0] >= args['prop'])])
            # write_check = 0
            # for score in roc_score:
                # if score>= args['prop']:
                    # write_check =1
                    # break
            # if write_check ==1:

                # f.write('{},{}\n'.format(smiles, ','.join([str(round(float(score),3)) for score in roc_score])))
            # count += 1
            torch.cuda.empty_cache()
    if args['reg']:
        df = pd.DataFrame(out, columns=["predict_value"])
        dff = pd.concat([data, df], axis=1)
        dff.to_csv(args['output'], index=False)
        sorted_dff = dff.sort_values(by='predict_value', ascending=False)
        dff_top_k = sorted_dff.head(args['top_k'])
        dff_top_k.to_csv(top_k_file, index=False)
        dff_top_k.to_csv(top_k_file, index=False)
    else:
        df = pd.DataFrame(out, columns=["predict_value", "predict_label"])
        print('screen precent ', round((sum(np.array(list(df["predict_label"]))) / len(df)), 2))
        dff = pd.concat([data, df], axis=1)
        dff.to_csv(args['output'], index=False)
        print("result file save in : <", args['output'], ">")
        filtered_dff = dff[dff['predict_label'] == 1]
        sorted_dff = filtered_dff.sort_values(by='predict_value', ascending=False)
        dff_top_k = sorted_dff.head(args['top_k'])
        dff_top_k.to_csv(top_k_file, index=False)
        # f.close()
def screen(file='', sep=',', models=None,prop=0.5, smiles_col='Smiles',out_file=None,tasks=1, opt_res=None, top_k=10, task_type="cla"):

    my_df = pd.read_csv(file, engine='python', sep=sep)
    if "label" not in list(my_df.columns):
        my_df = my_df.reset_index()  
        my_df = my_df.rename(columns={'index': 'label'})
    
    device = torch.device("cpu")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    reg = True if task_type == 'reg' else False
    args = {'device': device, 'metric': 'roc_auc','prop':prop,'data':my_df,'smiles_col':smiles_col,'tasks':tasks,'top_k':top_k,'reg': reg}
    # outputs = os.path.join(out_dir,
                           # file.split('/')[-1].replace('.csv', '_screen_{}_{}.csv'.format(args['prop'], 'DNN')))
    outputs = out_file
    if os.path.exists(outputs):
        print(outputs, 'has done')
    else:
        args['output'] = outputs
        FP_type = models.split('/')[-1].split('_')[1]
        model_name = models.split('/')[-1].split('_')[0]
        args['model_name'] = model_name
        # model_dir = out_dir.replace(out_dir.split('/')[-1],'model_save')
        model_dir = os.path.dirname(os.path.dirname(models))
        print(smiles_col)
        # data_x, data_y = create_des(my_df[smiles_col], list(range(len(my_df))), FP_type=FP_type, model_dir=model_dir)
        # print(data_x)
        # dataset = MyDataset(data_x, data_y)
        # loader = DataLoader(dataset,  collate_fn=collate_fn)
        AtomFeaturizer = AttentiveFPAtomFeaturizer
        BondFeaturizer = AttentiveFPBondFeaturizer
        dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(my_df.iloc[:, :], smiles_to_bigraph, AtomFeaturizer,
                                                                    BondFeaturizer, smiles_col,
                                                                    file.replace('.csv', '.bin'))
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_molgraphs, num_workers=0)
        
        # inputs = data_x.shape[1]
        # hideen_unit = (eval(models.split('/')[-1].split('_')[5]),
                       # eval(models.split('/')[-1].split('_')[6])
                       # ,eval(models.split('/')[-1].split('_')[7]))

        # dropout = eval(models.split('/')[-1].split('_')[4])
        if model_name  == 'gcn':
            best_model = GCNClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                   gcn_hidden_feats=opt_res['gcn_hidden_feats'],
                                   n_tasks=1,
                                   classifier_hidden_feats=opt_res['classifier_hidden_feats'])
        elif model_name == 'gat':
            best_model = GATClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                   gat_hidden_feats=opt_res['gat_hidden_feats'],
                                   num_heads=opt_res['num_heads'], n_tasks=1,
                                   classifier_hidden_feats=opt_res['classifier_hidden_feats'])
        elif model_name == 'attentivefp':
            best_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'),
                                 edge_feat_size=BondFeaturizer.feat_size('e'),
                                 num_layers=opt_res['num_layers'],
                                 num_timesteps=opt_res['num_timesteps'],
                                 graph_feat_size=opt_res['graph_feat_size'], output_size=1,
                                 dropout=opt_res['dropout'])
        elif model_name == 'mpnn':
            best_model = MPNNModel(node_input_dim=AtomFeaturizer.feat_size('h'),
                               edge_input_dim=BondFeaturizer.feat_size('e'),
                               output_dim=1, node_hidden_dim=opt_res['node_hidden_dim'],
                               edge_hidden_dim=opt_res['edge_hidden_dim'],
                               num_layer_set2set=opt_res['num_layer_set2set'])
        # best_model = MyDNN(inputs=inputs, hideen_units=hideen_unit, outputs=tasks,
                           # dp_ratio=dropout, reg=False)
        best_model.load_state_dict(torch.load(models, map_location=device)['model_state_dict'])
        best_model.to(device)
        run_an_eval_epoch(best_model, loader, args)
#
# screen(models ='/data/jianping/bokey/OCAICM/dataset/chembl31/model_save/DNN/cla_ECFP4_cluster_dataset_0.0016_64_512_128_0.0047_early_stop.pth',
#        file='/data/jianping/bokey/OCAICM/dataset/chembl31/model_save/BACE_screen_0.5_DNN.csv',prop = 0.5,sep = ',',
#        out_dir='/data/jianping/bokey/OCAICM/dataset/chembl31/model_save',smiles_col='cano_smiles',tasks=1)