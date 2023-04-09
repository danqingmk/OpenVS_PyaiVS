import time
import joblib
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import warnings
from PyaiVS.splitdater import split_dataset
from PyaiVS.feature_create import create_des
from PyaiVS.data_utils import TVT,statistical
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, mean_squared_error, \
    r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import os
np.random.seed(43)
def all_one_zeros(data):
    if (len(np.unique(data)) == 2):
        flag = False
    else:
        flag = True
    return flag



start = time.time()
warnings.filterwarnings("ignore")

# the metrics for classification

def best_model_runing(seed,best_hyper,data,split_type='random',FP_type='ECFP4',task_type='cla',model_dir = False):
    pd_res = []
    if task_type == 'cla':
        while True:
            # data_tr_x, data_va_x, data_te_x, data_tr_y, data_va_y, data_te_y = split_dataset(X, Y,
            #                                                                                      split_type=split_type,
            #                                                                                      valid_need=True,
            #                                                                                      random_state=seed)
            data_x, data_te_x, data_y, data_te_y = data.set2ten(seed)
            data_tr_x, data_va_x, data_tr_y, data_va_y = split_dataset(data_x, data_y, split_type='random', valid_need=False,
                                                                       random_state=seed, train_size=(8 / 9))
            data_tr_x, data_tr_y = create_des(data_tr_x, data_tr_y, FP_type=FP_type,model_dir=model_dir)
            data_va_x, data_va_y = create_des(data_va_x, data_va_y, FP_type=FP_type,model_dir=model_dir)
            data_te_x, data_te_y = create_des(data_te_x, data_te_y, FP_type=FP_type,model_dir=model_dir)

            if (all_one_zeros(data_tr_y) or all_one_zeros(data_va_y) or all_one_zeros(data_te_y)):
                # print(
                #     '\ninvalid random seed {} due to one class presented in the {} splitted sets...'.format(seed,
                #                                                                                             split_type))
                # print('Changing to another random seed...\n')
                seed = np.random.randint(50, 999999)
            else:

                break
    else :
        data_x, data_te_x, data_y, data_te_y = data.set2ten(seed)
        data_tr_x, data_va_x, data_tr_y, data_va_y = split_dataset(data_x, data_y, split_type=split_type,
                                                                   valid_need=False,
                                                                   random_state=seed, train_size=(8 / 9))
        data_tr_x, data_tr_y = create_des(data_tr_x, data_tr_y, FP_type=FP_type)
        data_va_x, data_va_y = create_des(data_va_x, data_va_y, FP_type=FP_type)
        data_te_x, data_te_y = create_des(data_te_x, data_te_y, FP_type=FP_type)

    model = KNeighborsClassifier(n_neighbors=best_hyper['n_neighbors'],leaf_size=best_hyper['leaf_size'],weights='uniform', algorithm='auto', p=2,
                                     metric='minkowski', metric_params=None, n_jobs=None) \
        if task_type == 'cla' else KNeighborsRegressor(n_neighbors=best_hyper['n_neighbors'],
                                      leaf_size=best_hyper['leaf_size'], weights='uniform', algorithm='auto', p=2, metric='minkowski',
                                 metric_params=None, n_jobs=None)

    model.fit(data_tr_x, data_tr_y)
    if model_dir :
        model_name = str(model_dir) +'/%s_%s_%s_%s_%s'%(split_type,task_type,FP_type,seed,'KNN_bestModel.pkl')
        joblib.dump(model,model_name)
    num_of_compounds = data_x.shape[0]+data_te_x.shape[0]
    if task_type == 'cla':
        tr_pred = model.predict_proba(data_tr_x)
        tr_results = [seed, FP_type, split_type, 'tr', num_of_compounds]
        tr_results.extend(statistical(data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))
        pd_res.append(tr_results)
        # validation set
        va_pred = model.predict_proba(data_va_x)
        va_results = [seed, FP_type, split_type, 'va', num_of_compounds]
        va_results.extend(statistical(data_va_y, np.argmax(va_pred, axis=1), va_pred[:, 1]))
        pd_res.append(va_results)
        # test set
        te_pred = model.predict_proba(data_te_x)
        te_results = [seed, FP_type, split_type, 'te', num_of_compounds]
        te_results.extend(statistical(data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
        pd_res.append(te_results)
    else:
        # training set
        tr_pred = model.predict(data_tr_x)
        tr_results = [seed, FP_type, split_type, 'tr',  num_of_compounds,
                      np.sqrt(mean_squared_error(data_tr_y, tr_pred)), r2_score(data_tr_y, tr_pred),
                      mean_absolute_error(data_tr_y, tr_pred)]
        pd_res.append(tr_results)
        # validation set
        va_pred = model.predict(data_va_x)
        va_results = [seed, FP_type, split_type, 'va',  num_of_compounds,
                      np.sqrt(mean_squared_error(data_va_y, va_pred)), r2_score(data_va_y, va_pred),
                      mean_absolute_error(data_va_y, va_pred)]
        pd_res.append(va_results)
        # test set
        te_pred = model.predict(data_te_x)
        te_results = [seed, FP_type, split_type, 'te',  num_of_compounds,
                      np.sqrt(mean_squared_error(data_te_y, te_pred)), r2_score(data_te_y, te_pred),
                      mean_absolute_error(data_te_y, te_pred)]
        pd_res.append(te_results)

    return pd_res
def tvt_knn(X,Y,split_type='random',FP_type='ECFP4',task_type='cla',model_dir=False):
    random_state = 42
    while True:

        # data_tr_x, data_va_x, data_te_x, data_tr_y, data_va_y, data_te_y = split_dataset(X, Y,
        #                                                                                  split_type=split_type,
        #                                                                                  valid_need=True,
        #                                                                                  random_state=random_state)
        data = TVT(X, Y)
        data_x, data_te_x, data_y, data_te_y = data.set2ten(0)
        
        data_tr_x, data_va_x, data_tr_y, data_va_y = split_dataset(data_x, data_y, split_type=split_type, valid_need=False,
                                                                   random_state=random_state, train_size=(8 / 9))

        data_tr_x, data_tr_y = create_des(data_tr_x, data_tr_y, FP_type=FP_type, model_dir=model_dir)
        data_va_x, data_va_y = create_des(data_va_x, data_va_y, FP_type=FP_type, model_dir=model_dir)
        data_te_x, data_te_y = create_des(data_te_x, data_te_y, FP_type=FP_type, model_dir=model_dir)

        if (all_one_zeros(data_tr_y) or all_one_zeros(data_va_y) or all_one_zeros(data_te_y)):
            # print(
            #     '\ninvalid random seed {} due to one class presented in the {} splitted sets...'.format('None',
            #                                                                                             split_type))
            #
            random_state += np.random.randint(50, 999999)
            #
            # print('Changing to another random seed {}\n'.format(random_state))
        else:


            break

    pd_res = []
    OPT_ITERS = 50

    space_ = {'n_neighbors': hp.choice('n_neighbors', np.arange(1, 20, 2).tolist()),
              'leaf_size': hp.choice('leaf_size', np.arange(1, 20, 2).tolist()),

              }
    n_neighbors_ls = np.arange(1, 20, 2).tolist()
    leaf_size_ls = np.arange(1, 20, 2).tolist()
    trials = Trials()
    def hyper_opt(args):
        model = KNeighborsClassifier(**args,weights='uniform', algorithm='auto', p=2, metric='minkowski', metric_params=None, n_jobs=None)\
            if task_type == 'cla' else KNeighborsRegressor(**args,weights='uniform', algorithm='auto', p=2, metric='minkowski', metric_params=None, n_jobs=None)

        model.fit(data_tr_x, data_tr_y)

        val_preds = model.predict_proba(data_va_x) if task_type == 'cla' else model.predict(data_va_x)
        loss = 1 - roc_auc_score(data_va_y, val_preds[:, 1]) if task_type == 'cla' else np.sqrt(
            mean_squared_error(data_va_y, val_preds))
        return {'loss': loss, 'status': STATUS_OK}


    # start hyper-parameters optimization
    best_results = fmin(hyper_opt, space_, algo=tpe.suggest, max_evals=OPT_ITERS, trials=trials, show_progressbar=False)


    para_file = str(model_dir).replace('model_save', 'param_save') + '/%s_%s_%s_%s' % (
    split_type, task_type, FP_type, 'KNN.param')
    if not os.path.exists(str(model_dir).replace('model_save', 'param_save')):
        os.makedirs(str(model_dir).replace('model_save', 'param_save'))
    print(os.path.exists(str(model_dir).replace('model_save', 'param_save')))
    f = open(para_file, 'w')
    f.write('%s' % {'n_neighbors':n_neighbors_ls[best_results['n_neighbors']],'leaf_size':leaf_size_ls[best_results['leaf_size']]})
    f.close()
    # best_model = KNeighborsClassifier(n_neighbors=n_neighbors_ls[best_results['n_neighbors']],
    #                                   leaf_size=leaf_size_ls[best_results['leaf_size']], weights='uniform', algorithm='auto', p=2, metric='minkowski',
    #                              metric_params=None, n_jobs=None)\
    #     if task_type == 'cla' else KNeighborsRegressor(n_neighbors=n_neighbors_ls[best_results['n_neighbors']],
    #                                   leaf_size=leaf_size_ls[best_results['leaf_size']], weights='uniform', algorithm='auto', p=2, metric='minkowski',
    #                              metric_params=None, n_jobs=None)
    # best_model.fit(data_tr_x, data_tr_y)
    # if model_dir :
    #     model_name = str(model_dir) +'/%s_%s_%s_%s'%(split_type,task_type,FP_type,'KNN_bestModel.pkl')
    #     joblib.dump(best_model,model_name)
    # num_of_compounds = len(X)
    # if task_type == 'cla':
    #     tr_pred = best_model.predict_proba(data_tr_x)
    #     tr_results = [FP_type, split_type, 'tr', num_of_compounds,
    #                   n_neighbors_ls[best_results['n_neighbors']],
    #                   leaf_size_ls[best_results['leaf_size']]]
    #     tr_results.extend(statistical(data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))
    #     pd_res.append(tr_results)
    #     # validation set
    #     va_pred = best_model.predict_proba(data_va_x)
    #     va_results = [FP_type, split_type, 'va', num_of_compounds,
    #                   n_neighbors_ls[best_results['n_neighbors']],
    #                   leaf_size_ls[best_results['leaf_size']],]
    #     va_results.extend(statistical(data_va_y, np.argmax(va_pred, axis=1), va_pred[:, 1]))
    #     pd_res.append(va_results)
    #     # test set
    #     te_pred = best_model.predict_proba(data_te_x)
    #     te_results = [FP_type, split_type, 'te', num_of_compounds,
    #                   n_neighbors_ls[best_results['n_neighbors']],
    #                   leaf_size_ls[best_results['leaf_size']]]
    #     te_results.extend(statistical(data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
    #     pd_res.append(te_results)
    #     para_res = pd.DataFrame(pd_res, columns=['FP_type', 'split_type', 'type','num_of_compounds',
    #            'n_neighbors', 'leaf_size', 'precision',
    #              'se', 'sp','acc', 'mcc', 'auc_prc', 'auc_roc'])
    # else:
    #     tr_pred = best_model.predict(data_tr_x)
    #     tr_results = [FP_type, split_type, 'tr', num_of_compounds,
    #                   n_neighbors_ls[best_results['n_neighbors']],
    #                   leaf_size_ls[best_results['leaf_size']],np.sqrt(mean_squared_error(data_tr_y, tr_pred)), r2_score(data_tr_y, tr_pred),
    #                   mean_absolute_error(data_tr_y, tr_pred)]
    #
    #     pd_res.append(tr_results)
    #     # validation set
    #     va_pred = best_model.predict(data_va_x)
    #     va_results = [FP_type, split_type, 'va', num_of_compounds,
    #                   n_neighbors_ls[best_results['n_neighbors']],
    #                   leaf_size_ls[best_results['leaf_size']],np.sqrt(mean_squared_error(data_va_y, va_pred)), r2_score(data_tr_y, tr_pred),
    #                   mean_absolute_error(data_va_y, va_pred)]
    #
    #     pd_res.append(va_results)
    #     # test set
    #     te_pred = best_model.predict(data_te_x)
    #     te_results = [FP_type, split_type, 'te', num_of_compounds,
    #                   n_neighbors_ls[best_results['n_neighbors']],
    #                   leaf_size_ls[best_results['leaf_size']],np.sqrt(mean_squared_error(data_te_y, te_pred)), r2_score(data_te_y, te_pred),
    #                   mean_absolute_error(data_te_y, te_pred)]
    #
    #     pd_res.append(te_results)
    #
    #     para_res = pd.DataFrame(pd_res, columns=['FP_type', 'split_type', 'type', 'num_of_compounds',
    #                                              'n_neighbors', 'leaf_size','rmse', 'r2', 'mae'])
    #
    #
    # pd_res = []
    # n_neighbors = n_neighbors_ls[best_results['n_neighbors']]
    # leaf_size = leaf_size_ls[best_results['leaf_size']]
    # best_results = {'n_neighbors':n_neighbors,'leaf_size':leaf_size}
    # for i in range(9):
    #     item = best_model_runing((i+1),best_results,data,split_type=split_type,FP_type=FP_type,task_type=task_type,model_dir=model_dir)
    #     pd_res.extend(item)
    #
    # if task_type == 'cla':
    #     best_res = pd.DataFrame(pd_res, columns=['seed', 'FP_type', 'split_type', 'type',
    #                                              'num_of_compounds', 'precision', 'se', 'sp',
    #                                              'acc', 'mcc', 'auc_prc', 'auc_roc'])
    #     pd1 = para_res[['FP_type', 'split_type', 'type',
    #                     'num_of_compounds', 'precision', 'se', 'sp',
    #                                              'acc', 'mcc', 'auc_prc', 'auc_roc']]
    #     pd1['seed'] = 0
    #     best_res = pd.concat([pd1, best_res], ignore_index=True)
    #     # best_res = best_res.groupby(['FP_type', 'split_type', 'type'])['acc', 'mcc', 'auc_prc', 'auc_roc'].mean()
    # else:
    #     best_res = pd.DataFrame(pd_res, columns=['seed', 'FP_type', 'split_type', 'type',
    #                                              'num_of_compounds', 'rmse', 'r2', 'mae'])
    #     pd1 = para_res[['FP_type', 'split_type', 'type',
    #                     'num_of_compounds', 'rmse', 'r2', 'mae']]
    #     pd1['seed'] = 0
    #     best_res = pd.concat([pd1, best_res], ignore_index=True)
    # return  para_res,best_res
def para_knn(X,Y,args=None,split_type='random',FP_type='ECFP4',task_type = 'cla',model_dir=None):
    param_file = str(model_dir).replace('model_save', 'param_save') + '/%s_%s_%s_%s' % (
    split_type, task_type, FP_type, 'KNN.param')
    args = eval(open(param_file, 'r').readline().strip()) if args == None else args
    data = TVT(X, Y)
    data_x, data_te_x, data_y, data_te_y = data.set2ten(0)
    data_tr_x, data_va_x, data_tr_y, data_va_y = split_dataset(data_x, data_y, split_type=split_type, valid_need=False,
                                                               random_state=42, train_size=(8 / 9))
    data_tr_x, data_tr_y = create_des(data_tr_x, data_tr_y, FP_type=FP_type, model_dir=model_dir)
    data_va_x, data_va_y = create_des(data_va_x, data_va_y, FP_type=FP_type, model_dir=model_dir)
    data_te_x, data_te_y = create_des(data_te_x, data_te_y, FP_type=FP_type, model_dir=model_dir)

    pd_res = []
    special_n = args['n_neighbors']
    special_ls = args['leaf_size']
    best_model = KNeighborsClassifier(n_neighbors=special_n,
                                      leaf_size=special_ls, weights='uniform', algorithm='auto', p=2, metric='minkowski',
                                 metric_params=None, n_jobs=None)\
        if task_type == 'cla' else KNeighborsRegressor(n_neighbors=special_n,
                                      leaf_size=special_ls, weights='uniform', algorithm='auto', p=2, metric='minkowski',
                                 metric_params=None, n_jobs=None)
    best_model.fit(data_tr_x, data_tr_y)
    if model_dir :
        model_name = str(model_dir) +'/%s_%s_%s_%s'%(split_type,task_type,FP_type,'KNN_bestModel.pkl')
        joblib.dump(best_model,model_name)
    num_of_compounds = len(X)
    if task_type == 'cla':
        tr_pred = best_model.predict_proba(data_tr_x)
        tr_results = [FP_type, split_type, 'tr', num_of_compounds,
                      special_n,
                      special_ls]
        tr_results.extend(statistical(data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))
        pd_res.append(tr_results)
        # validation set
        va_pred = best_model.predict_proba(data_va_x)
        va_results = [FP_type, split_type, 'va', num_of_compounds,
                      special_n,
                      special_ls]
        va_results.extend(statistical(data_va_y, np.argmax(va_pred, axis=1), va_pred[:, 1]))
        pd_res.append(va_results)
        # test set
        te_pred = best_model.predict_proba(data_te_x)
        te_results = [FP_type, split_type, 'te', num_of_compounds,
                      special_n,
                      special_ls]
        te_results.extend(statistical(data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
        pd_res.append(te_results)
        para_res = pd.DataFrame(pd_res, columns=['FP_type', 'split_type', 'type', 'num_of_compounds',
                                                 'para1', 'para2',
                                                 'precision', 'se', 'sp',
                                                 'acc', 'mcc', 'auc_prc', 'auc_roc'])
    else:
        # training set
        tr_pred = best_model.predict(data_tr_x)
        tr_results = [FP_type, split_type, 'tr',  num_of_compounds,
                      special_n,
                      special_ls,
                      np.sqrt(mean_squared_error(data_tr_y, tr_pred)), r2_score(data_tr_y, tr_pred),
                      mean_absolute_error(data_tr_y, tr_pred)]
        pd_res.append(tr_results)
        # validation set
        va_pred = best_model.predict(data_va_x)
        va_results = [FP_type, split_type,'va',  num_of_compounds,
                      special_n,
                      special_ls,
                      np.sqrt(mean_squared_error(data_va_y, va_pred)), r2_score(data_va_y, va_pred),
                      mean_absolute_error(data_va_y, va_pred)]
        pd_res.append(va_results)
        # test set
        te_pred = best_model.predict(data_te_x)
        te_results = [FP_type, split_type, 'te',  num_of_compounds,
                      special_n,
                      special_ls,
                      np.sqrt(mean_squared_error(data_te_y, te_pred)), r2_score(data_te_y, te_pred),
                      mean_absolute_error(data_te_y, te_pred)]
        pd_res.append(te_results)
        para_res = pd.DataFrame(pd_res, columns=['FP_type', 'split_type', 'type', 'num_of_compounds',
                                                 'C', 'gamma',
                                                 'rmse', 'r2', 'mae'])
    pd_res = []
    for i in range(9):
        item = best_model_runing((i + 1), args, data, split_type=split_type, FP_type=FP_type,
                                     task_type=task_type, model_dir=model_dir)
        pd_res.extend(item)

    if task_type == 'cla':
        best_res = pd.DataFrame(pd_res, columns=['seed', 'FP_type', 'split_type', 'type',
                                                     'num_of_compounds', 'precision', 'se', 'sp',
                                                     'acc', 'mcc', 'auc_prc', 'auc_roc'])
        pd1 = para_res[['FP_type', 'split_type', 'type',
                            'num_of_compounds', 'precision', 'se', 'sp',
                            'acc', 'mcc', 'auc_prc', 'auc_roc']]
        pd1['seed'] = 0
        best_res = pd.concat([pd1, best_res], ignore_index=True)
        # best_res = best_res.groupby(['FP_type', 'split_type', 'type'])['acc', 'mcc', 'auc_prc', 'auc_roc'].mean()
    else:
        best_res = pd.DataFrame(pd_res, columns=['seed', 'FP_type', 'split_type', 'type',
                                                     'num_of_compounds', 'rmse', 'r2', 'mae'])
        pd1 = para_res[['FP_type', 'split_type', 'type',
                            'num_of_compounds', 'rmse', 'r2', 'mae']]
        pd1['seed'] = 0
        best_res = pd.concat([pd1, best_res], ignore_index=True)
    result_dir = model_dir.replace('model_save', 'result_save')
    para_name, best_name = os.path.join(result_dir,
                                        '_'.join([split_type, 'KNN', FP_type, 'para.csv'])), os.path.join(
        result_dir, '_'.join([split_type, 'KNN', FP_type, 'best.csv']))
    para_res.to_csv(para_name, index=False)
    best_res.to_csv(best_name, index=False)
# df = pd.read_csv('/data/jianping/bokey/OCAICM/dataset/aurorab/aurorab.csv')
# args = {'n_neighbors':5,'leaf_size':17}
# a,b =para_knn(df['Smiles'],df['activity'],args,split_type='scaffold',FP_type='MACCS',task_type = 'cla',model_dir=False)
# # a,b =tvt_knn(df['Smiles'],df['activity'],split_type='scaffold',FP_type='MACCS',task_type = 'cla',model_dir=False)
# print(a,b)
# #