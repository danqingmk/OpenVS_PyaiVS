import pandas as pd
import os


dataset = 'chembl31'
record_file = '/data/jianping/bokey/OCAICM/dataset/{0}/{0}'.format(dataset)+"_record_max.csv"
print(record_file)
f = open(record_file,'w')
f.write('model;split;des;acc;mcc;auc_roc;param_file;model_file\n')
path = '/data/jianping/bokey/OCAICM/dataset/chembl31/{}/'.format(dataset)
result = os.path.join(path,'result_save')
models = os.listdir(result)
data = pd.DataFrame()
for model in ['SVM','XGB','RF','KNN']:
    files = [os.path.join(os.path.join(result,model,file)) for file in os.listdir(os.path.join(result,model)) if 'best.csv' in file]
    for file in files:
        df = pd.read_csv(file)
        df=df[df['type']=='te']
        # df['diff_mcc'] = abs(df['mcc']-sum(df['mcc'])/len(df))
        # df =df.sort_values('diff_mcc')
        df = df.sort_values('mcc',ascending=False)
        df = df.reset_index()
        model_path = os.path.join(path,'model_save')
        param_path = os.path.join(path,'param_save')
        model_info =[df.loc[0,'split_type'], 'cla', df.loc[0,'FP_type'], model, 'bestModel.pkl']
        param_info = [df.loc[0,'split_type'], 'cla', df.loc[0,'FP_type'], model]
        if df.loc[0,'seed'] ==0:
            pass
        else:
            model_info.insert(3,str(df.loc[0,'seed']))
        model_file = os.path.join(os.path.join(model_path,model),'_'.join(model_info))
        param_file = os.path.join(os.path.join(param_path,model),'_'.join(param_info))+'.param'
        f.write('%s;%s;%s;%s;%s;%s;%s;%s\n'%(model,df.loc[0,'split_type'],df.loc[0,'FP_type'],df.loc[0,'acc'],df.loc[0,'mcc'],sum(df['auc_roc'])/len(df),param_file,model_file))


for model in ['DNN']:
    files = [os.path.join(os.path.join(result, model, file)) for file in os.listdir(os.path.join(result, model)) if
             'best.csv' in file]
    for file in files:
        df = pd.read_csv(file)
        df = df[df['type'] == 'te']
        # df['diff_mcc'] = abs(df['mcc'] - sum(df['mcc']) / len(df))
        # df = df.sort_values('diff_mcc')
        df = df.sort_values('mcc', ascending=False)
        df = df.reset_index()
        model_path = os.path.join(path,'model_save')
        param_path = os.path.join(path,'param_save')
        param_info = [df.loc[0, 'split_type'], 'cla', df.loc[0, 'FP_type'], model]
        param_file = os.path.join(os.path.join(param_path, model), '_'.join(param_info)) + '.param'
        opt_res = eval(
            open(os.path.join(param_path, model, '{}_cla_{}_{}.param'.format(df.loc[0,'split_type'], df.loc[0,'FP_type'], model)), 'r').readline().strip())
        if df.loc[0, 'seed'] == 0:
            model_file = '%s/%s/%s_%s_%s_%s_%.4f_%d_%d_%d_%.4f_early_stop.pth' % (
                model_path,'DNN', 'cla', df.loc[0, 'FP_type'], df.loc[0, 'split_type'], 'dataset', opt_res['dropout'],
                opt_res['hidden_units'][0],
                opt_res['hidden_units'][1],
                opt_res['hidden_units'][2],
                opt_res['l2'])
        else:
            model_file = '%s/%s/%s_%s_%s_%s_%.4f_%d_%d_%d_%.4f_early_stop_%d.pth' % (
                model_path,'DNN', 'cla', df.loc[0,'FP_type'], df.loc[0,'split_type'], 'dataset', opt_res['dropout'],
                opt_res['hidden_units'][0],
                opt_res['hidden_units'][1],
                opt_res['hidden_units'][2],
                opt_res['l2'], df.loc[0, 'seed'])
        f.write('%s;%s;%s;%s;%s;%s;%s;%s\n'%(model,df.loc[0,'split_type'],df.loc[0,'FP_type'],df.loc[0,'acc'],df.loc[0,'mcc'],sum(df['auc_roc'])/len(df),param_file,model_file))


for model in ['gcn','gat','attentivefp']:
    files = [os.path.join(os.path.join(result, model, file)) for file in os.listdir(os.path.join(result, model)) if
             'best.csv' in file]
    for file in files:
        df = pd.read_csv(file)
        df = df[df['type'] == 'te']
        # df['diff_mcc'] = abs(df['mcc']-sum(df['mcc'])/len(df))
        # df =df.sort_values('diff_mcc')
        df = df.sort_values('mcc', ascending=False)
        df = df.reset_index()
        model_path = os.path.join(path,'model_save')
        param_path = os.path.join(path,'param_save')
        param_info = [df.loc[0,'split'], 'cla', model]
        param_file = os.path.join(os.path.join(param_path, model), '_'.join(param_info)) + '.param'
        param = open(param_file, 'r').readline().strip()
        if model == 'gcn':
            opt_res = eval(param)
            if df.loc[0, 'seed'] == 0:
                model_file = '%s/%s/%s_%s_%s_%s_%.6f_%s_%s.pth' % (model_path, model, model, df.loc[0,'split'], 'cla',
                                                                   opt_res['l2'], opt_res['lr'],
                                                                   opt_res['gcn_hidden_feats'],
                                                                   opt_res['classifier_hidden_feats'])
            else:
                model_file = '%s/%s/%s_%s_%s_%s_%.6f_%s_%s_%s.pth' % (model_path, model, model, df.loc[0, 'split'], 'cla',
                                                                   opt_res['l2'], opt_res['lr'],
                                                                   opt_res['gcn_hidden_feats'],
                                                                   opt_res['classifier_hidden_feats'],df.loc[0,'seed'])

        elif model == 'gat':
            opt_res = eval(param)
            if df.loc[0, 'seed'] == 0:
                model_file = '%s/%s/%s_%s_%s_%s_%.6f_%s_%s_%s.pth' % (model_path, model, model, df.loc[0,'split'], 'cla',
                                                                      opt_res['l2'], opt_res['lr'],
                                                                      opt_res['gat_hidden_feats'],
                                                                      opt_res['num_heads'],
                                                                      opt_res['classifier_hidden_feats'])
            else:
                model_file = '%s/%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (
                model_path, model, model, df.loc[0, 'split'], 'cla',
                opt_res['l2'], opt_res['lr'],
                opt_res['gat_hidden_feats'],
                opt_res['num_heads'],
                opt_res['classifier_hidden_feats'],df.loc[0,'seed'])


        else:
            opt_res = eval(param)
            if df.loc[0, 'seed'] == 0:
                model_file = '%s/%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (model_path, model, model, df.loc[0,'split'], 'cla',
                                                                         opt_res['l2'], opt_res['lr'],
                                                                         opt_res['num_layers'],
                                                                         opt_res['num_timesteps'],
                                                                         opt_res['graph_feat_size'],
                                                                         opt_res['dropout'])
            else:
                model_file = '%s/%s/%s_%s_%s_%s_%.6f_%s_%s_%s_%s_%s.pth' % (
                model_path, model, model, df.loc[0, 'split'], 'cla',
                opt_res['l2'], opt_res['lr'],
                opt_res['num_layers'],
                opt_res['num_timesteps'],
                opt_res['graph_feat_size'],
                opt_res['dropout'],df.loc[0,'seed'])
        f.write('%s;%s;%s;%s;%s;%s;%s;%s\n'%(model,df.loc[0,'split'],model,df.loc[0,'acc'],df.loc[0,'mcc'],sum(df['auc_roc'])/len(df),param_file,model_file))
f.close()





