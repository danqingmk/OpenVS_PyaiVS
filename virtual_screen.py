import os
import multiprocessing as mp
import pandas as pd
from PyaiVS import ml_screener
from PyaiVS import dl_screener
from PyaiVS import dnn_screener


def model_screen(model=None, FP=None, split=None, screen_file=None, prop=0.5, sep=',', model_dir=None, smiles_col=None):
    split = split if type(split) != type(None) else 'random'
    model = model if type(model) != type(None) else 'SVM'
    FP = FP if type(FP) != type(None) else 'MACCS'
    FP_list = ['2d-3d', 'MACCS', 'ECFP4', 'pubchem', 'gcn', 'gat', 'attentivefp', 'mpnn']
    split_list = ['random', 'scaffold', 'cluster']
    model_list = ['SVM', 'KNN', 'DNN', 'RF', 'XGB', 'gcn', 'gat', 'attentivefp', 'mpnn']
    assert len(set(list([split])) - set(split_list)) == 0, '{} element need in {}'.format(split, split_list)
    assert len(set(list([model])) - set(model_list)) == 0, '{} element need in {}'.format(model, model_list)
    assert len(set(list([FP])) - set(FP_list)) == 0, '{} element need in {}'.format(FP, FP_list)
    assert os.path.exists(model_dir), 'no such model_dir {}'.format(model_dir)
    assert len(set(list([screen_file.split('.')[-1]])) - set(
        list(['csv', 'txt', 'tsv']))) == 0, '{} need in ["csv","txt","tsv"]'.format(screen_file.split('.')[-1])
    out_dir = model_dir.replace('model_save', 'screen')
    if model in ['SVM', 'KNN', 'RF', 'XGB']:

        result_file = os.path.join(os.path.join(model_dir.replace('model', 'result'), model),
                                   '_'.join([split, model, FP, 'best.csv']))
        seq = pd.read_csv(result_file)
        seed = seq[seq['type'] == 'te'].sort_values('mcc', ascending=False)['seed'].tolist()[0]
        if seed == 0:
            model_file = '_'.join([split, 'cla', FP, model, 'bestModel.pkl'])
            model_path = os.path.join(os.path.join(model_dir, model), model_file)
            assert os.path.exists(model_path), 'no such model_path {}'.format(model_path)
        else:
            model_file = '_'.join([split, 'cla', FP, str(seed), model, 'bestModel.pkl'])
            model_path = os.path.join(os.path.join(model_dir, model), model_file)
            assert os.path.exists(model_path), 'no such model_path {}'.format(model_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if os.path.isdir(screen_file):
            p = mp.Pool(processes=4)
            for file_content in os.listdir(screen_file):
                file_path = os.path.join(screen_file, file_content)
                param = {'file': file_path, 'sep': sep, 'models': model_path, 'prop': prop, 'out_dir': out_dir}
                get = p.apply_async(ml_screener.cir_file, kwds=param)
            p.close()
            p.join()
        elif os.path.isfile(screen_file):
            print(screen_file, model_path, smiles_col)
            ml_screener.cir_file(file=screen_file, sep=sep, models=model_path, prop=prop, out_dir=out_dir,
                                 smiles_col=smiles_col)
        else:
            print('What\'s this ?')
    elif model == 'DNN':
        param_dir = model_dir.replace('model_save', 'param_save')
        param_file = os.path.join(os.path.join(param_dir, model), '_'.join([split, 'cla', FP, model + '.param']))
        parameter = eval(open(param_file, 'r').readline().strip())
        result_dir = model_dir.replace('model_save', 'result_save')
        result_file = os.path.join(os.path.join(result_dir, model), '_'.join([split, model, FP, 'best.csv']))
        seq = pd.read_csv(result_file)
        seed = seq[seq['type'] == 'te'].sort_values('mcc', ascending=False)['seed'].tolist()[0]
        if seed == 0:
            model_file = '_'.join(['cla', FP, split, "dataset", '%.4f' % parameter['dropout'],
                                   '_'.join([str(param) for param in parameter['hidden_units']]),
                                   '%.4f' % parameter['l2'], 'early_stop.pth'])
        else:
            model_file = '_'.join(['cla', FP, split, "dataset", '%.4f' % parameter['dropout'],
                                   '_'.join([str(param) for param in parameter['hidden_units']]),
                                   '%.4f' % parameter['l2'], 'early_stop_%d.pth' % seed])
        model_path = os.path.join(model_dir, model, model_file)
        assert os.path.exists(model_path), 'no such model_path {}'.format(model_path)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if os.path.isdir(screen_file):
            for file_content in os.listdir(screen_file):
                p = mp.Pool(processes=1)
                file_path = os.path.join(screen_file, file_content)
                param = {'file': file_path, 'sep': sep, 'models': model_path, 'prop': prop, 'out_dir': out_dir,
                         'smiles_col': smiles_col}
                get = p.apply_async(dnn_screener.screen, kwds=param)
                p.close()
                p.join()
        elif os.path.isfile(screen_file):
            dnn_screener.screen(file=screen_file, sep=sep, models=model_path, prop=prop, out_dir=out_dir,
                                smiles_col=smiles_col)
        else:
            print('What\'s this ?')
    else:
        param_dir = model_dir.replace('model_save', 'param_save')
        param_file = os.path.join(os.path.join(param_dir, model), '_'.join([split, 'cla', model + '.param']))
        parameter = eval(open(param_file, 'r').readline().strip())
        result_dir = model_dir.replace('model_save', 'result_save')
        result_file = os.path.join(os.path.join(result_dir, model), '_'.join([split, model, 'best.csv']))
        seq = pd.read_csv(result_file)
        seed = seq[seq['type'] == 'te'].sort_values('mcc', ascending=False)['seed'].tolist()[0]
        if seed == 0:
            if model == 'gcn':
                model_file = '_'.join([model, split, "cla", str(parameter['l2']), '%.6f' % parameter['lr'],
                                       str(parameter['gcn_hidden_feats']), str(parameter['classifier_hidden_feats']),
                                       '.pth'])
            elif model == 'gat':
                model_file = '_'.join([model, split, "cla", str(parameter['l2']), '%.6f' % parameter['lr'],
                                       str(parameter['gat_hidden_feats']), str(parameter['num_heads']),
                                       str(parameter['classifier_hidden_feats']), '.pth'])
            elif model == 'attentivefp':
                model_file = '_'.join([model, split, "cla", str(parameter['l2']), '%.6f' % parameter['lr'],
                                       str(parameter['num_layers']), str(parameter['num_timesteps']),
                                       str(parameter['graph_feat_size']), str(parameter['dropout']), '.pth'])
            elif model == 'mpnn':
                model_file = '_'.join([model, split, "cla", str(parameter['l2']), '%.6f' % parameter['lr'],
                                       str(parameter['node_hidden_dim']), str(parameter['edge_hidden_dim']),
                                       str(parameter['num_layer_set2set']), '.pth'])

        else:
            if model == 'gcn':
                model_file = '_'.join([model, split, "cla", str(parameter['l2']), '%.6f' % parameter['lr'],
                                       str(parameter['gcn_hidden_feats']), str(parameter['classifier_hidden_feats']),
                                       '%d.pth' % seed])
            elif model == 'gat':
                model_file = '_'.join([model, split, "cla", str(parameter['l2']), '%.6f' % parameter['lr'],
                                       str(parameter['gat_hidden_feats']), str(parameter['num_heads']),
                                       str(parameter['classifier_hidden_feats']), '%d.pth' % seed])
            elif model == 'attentivefp':
                model_file = '_'.join([model, split, "cla", str(parameter['l2']), '%.6f' % parameter['lr'],
                                       str(parameter['num_layers']), str(parameter['num_timesteps']),
                                       str(parameter['graph_feat_size']), str(parameter['dropout']), '%d.pth' % seed])
            elif model == 'mpnn':
                model_file = '_'.join([model, split, "cla", str(parameter['l2']), '%.6f' % parameter['lr'],
                                       str(parameter['node_hidden_dim']), str(parameter['edge_hidden_dim']),
                                       str(parameter['num_layer_set2set']), '%d.pth' % seed])
        model_path = os.path.join(model_dir, model, model_file)
        assert os.path.exists(model_path), 'no such model_path {}'.format(model_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if os.path.isdir(screen_file):
            for file_content in os.listdir(screen_file):
                p = mp.Pool(processes=1)
                file_path = os.path.join(screen_file, file_content)
                param = {'file': file_path, 'sep': sep, 'models': model_path, 'prop': prop, 'out_dir': out_dir,
                         'smiles_col': smiles_col}
                get = p.apply_async(dl_screener.screen, kwds=param)
                p.close()
                p.join()
        elif os.path.isfile(screen_file):
            print(out_dir, screen_file)
            dl_screener.screen(file=screen_file, sep=sep, models=model_path, prop=prop, out_dir=out_dir,
                               smiles_col=smiles_col)
        else:
            print('What\'s this ?')

# model_screen(model='mpnn',
#                            split='random',
#                            FP='ECFP4',
#                            prop=0.8,
#                            model_dir='/data/jianping/bokey/test/dataset/abcg2/model_save',
#                            screen_file='/data/jianping/bokey/OCAICM/dataset/a1/base.csv',
#                            sep=';',
#                            smiles_col='smiles')
