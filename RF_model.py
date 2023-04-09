from PyaiVS.data_utils import TVT,statistical
import warnings
from PyaiVS.splitdater import split_dataset
from PyaiVS.feature_create import create_des
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import joblib
from sklearn.metrics import roc_auc_score, mean_squared_error,r2_score, mean_absolute_error

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
def all_one_zeros(data):
    if (len(np.unique(data)) == 2):
        flag = False
    else:
        flag = True
    return flag


warnings.filterwarnings("ignore")

# the metrics for classification

def best_model_runing(seed,best_hyper,data,split_type='random',FP_type='ECFP4',task_type ='cla',model_dir=False):
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
    else:

        data_x, data_te_x, data_y, data_te_y = data.set2ten(seed)
        data_tr_x, data_va_x, data_tr_y, data_va_y = split_dataset(data_x, data_y, split_type='random',
                                                                   valid_need=False,
                                                                   random_state=seed, train_size=(8 / 9))
        data_tr_x, data_tr_y = create_des(data_tr_x, data_tr_y, FP_type=FP_type)
        data_va_x, data_va_y = create_des(data_va_x, data_va_y, FP_type=FP_type)
        data_te_x, data_te_y = create_des(data_te_x, data_te_y, FP_type=FP_type)

    model = RandomForestClassifier(n_estimators= best_hyper['n_estimators'],
                                   max_depth=best_hyper['max_depth'],
                                   min_samples_leaf=best_hyper['min_samples_leaf'],
                                   max_features=best_hyper['max_features'],

                                   n_jobs=6, random_state=1, verbose=0, class_weight='balanced') \
        if task_type == 'cla' else RandomForestRegressor(
        n_estimators=best_hyper['n_estimators'],
        max_depth=best_hyper['max_depth'],
        min_samples_leaf=best_hyper['min_samples_leaf'],
        max_features=best_hyper['max_features'],

        n_jobs=6, random_state=1, verbose=0)

    model.fit(data_tr_x, data_tr_y)
    if model_dir :
        model_name = str(model_dir) +'/%s_%s_%s_%s_%s'%(split_type,task_type,FP_type,seed,'RF_bestModel.pkl')
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
        tr_results = [seed, FP_type, split_type, 'tr', num_of_compounds,
                      np.sqrt(mean_squared_error(data_tr_y, tr_pred)), r2_score(data_tr_y, tr_pred),
                      mean_absolute_error(data_tr_y, tr_pred)]
        pd_res.append(tr_results)
        # validation set
        va_pred = model.predict(data_va_x)
        va_results = [seed, FP_type, split_type, 'va', num_of_compounds,
                      np.sqrt(mean_squared_error(data_va_y, va_pred)), r2_score(data_va_y, va_pred),
                      mean_absolute_error(data_va_y, va_pred)]
        pd_res.append(va_results)
        # test set
        te_pred = model.predict(data_te_x)
        te_results = [seed, FP_type, split_type, 'te', num_of_compounds,
                      np.sqrt(mean_squared_error(data_te_y, te_pred)), r2_score(data_te_y, te_pred),
                      mean_absolute_error(data_te_y, te_pred)]
        pd_res.append(te_results)

    return pd_res
def tvt_rf(X,Y,split_type='random',FP_type='ECFP4',task_type='cla',model_dir=False):
    random_state = 42
    while True:

        # data_tr_x, data_va_x, data_te_x, data_tr_y, data_va_y, data_te_y = split_dataset(X, Y,
        #                                                                                  split_type=split_type,
        #                                                                                  valid_need=True,
        #                                                                                  random_state=random_state)
        data = TVT(X,Y)
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

    space_ = {'n_estimators': hp.choice('n_estimators', [10, 50, 100, 200, 300, 400, 500]),
              'max_depth': hp.choice('max_depth', range(3, 12)),
              'min_samples_leaf': hp.choice('min_samples_leaf', [1, 3, 5, 10, 20, 50]),

              'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.7, 0.8, 0.9])
              }
    n_estimators_ls = [10, 50, 100, 200, 300, 400, 500]
    max_depth_ls = range(3, 12)
    min_samples_leaf_ls = [1, 3, 5, 10, 20, 50]
    max_features_ls = ['sqrt', 'log2', 0.7, 0.8, 0.9]

    trials = Trials()

    def hyper_opt(args):
        model = RandomForestClassifier(**args, n_jobs=6, random_state=1, verbose=0, class_weight='balanced') \
            if task_type == 'cla' else RandomForestRegressor(**args, n_jobs=6, random_state=1, verbose=0)
        model.fit(data_tr_x, data_tr_y)
        val_preds = model.predict_proba(data_va_x) if task_type == 'cla' else model.predict(data_va_x)
        loss = 1 - roc_auc_score(data_va_y, val_preds[:, 1]) if task_type == 'cla' else np.sqrt(
            mean_squared_error(data_va_y, val_preds))
        return {'loss': loss, 'status': STATUS_OK}


    # start hyper-parameters optimization
    best_results = fmin(hyper_opt, space_, algo=tpe.suggest, max_evals=OPT_ITERS, trials=trials, show_progressbar=False)

    # best_model = RandomForestClassifier(n_estimators=n_estimators_ls[best_results['n_estimators']],
    #                                     max_depth=max_depth_ls[best_results['max_depth']],
    #                                     min_samples_leaf=min_samples_leaf_ls[best_results['min_samples_leaf']],
    #                                     max_features=max_features_ls[best_results['max_features']],
    #
    #                                     n_jobs=6, random_state=1, verbose=0, class_weight='balanced') \
    #     if task_type == 'cla' else RandomForestRegressor(
    #     n_estimators=n_estimators_ls[best_results['n_estimators']],
    #     max_depth=max_depth_ls[best_results['max_depth']],
    #     min_samples_leaf=min_samples_leaf_ls[best_results['min_samples_leaf']],
    #     max_features=max_features_ls[best_results['max_features']],
    #
    #     n_jobs=6, random_state=1, verbose=0)
    # best_model.fit(data_tr_x, data_tr_y)
    # if model_dir :
    #     model_name = str(model_dir) +'/%s_%s_%s_%s'%(split_type,task_type,FP_type,'RF_bestModel.pkl')
    #     joblib.dump(best_model,model_name)
    # num_of_compounds = len(X)
    # if task_type == 'cla':
    #     tr_pred = best_model.predict_proba(data_tr_x)
    #     tr_results = [FP_type, split_type, 'tr', num_of_compounds,
    #                   n_estimators_ls[best_results['n_estimators']],
    #                   max_depth_ls[best_results['max_depth']],
    #                   min_samples_leaf_ls[best_results['min_samples_leaf']],
    #
    #                   max_features_ls[best_results['max_features']]]
    #     tr_results.extend(statistical(data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))
    #     pd_res.append(tr_results)
    #     # validation set
    #     va_pred = best_model.predict_proba(data_va_x)
    #     va_results = [FP_type, split_type, 'va', num_of_compounds,
    #                   n_estimators_ls[best_results['n_estimators']],
    #                    max_depth_ls[best_results['max_depth']],
    #                    min_samples_leaf_ls[best_results['min_samples_leaf']],
    #
    #                    max_features_ls[best_results['max_features']]]
    #     va_results.extend(statistical(data_va_y, np.argmax(va_pred, axis=1), va_pred[:, 1]))
    #     pd_res.append(va_results)
    #     # test set
    #     te_pred = best_model.predict_proba(data_te_x)
    #     te_results = [FP_type, split_type, 'te', num_of_compounds,
    #                   n_estimators_ls[best_results['n_estimators']],
    #                    max_depth_ls[best_results['max_depth']],
    #                    min_samples_leaf_ls[best_results['min_samples_leaf']],
    #
    #                    max_features_ls[best_results['max_features']]]
    #     te_results.extend(statistical(data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
    #     pd_res.append(te_results)
    #     para_res = pd.DataFrame(pd_res, columns=['FP_type', 'split_type', 'type','num_of_compounds',
    #             'n_estimators', 'max_depth','min_samples_leaf','max_features',
    #             'precision', 'se', 'sp',
    #             'acc', 'mcc', 'auc_prc', 'auc_roc'])
    # else:
    #     # training set
    #     tr_pred = best_model.predict(data_tr_x)
    #     tr_results = [FP_type, split_type, 'va', num_of_compounds,
    #                   n_estimators_ls[best_results['n_estimators']],
    #                   max_depth_ls[best_results['max_depth']],
    #                   min_samples_leaf_ls[best_results['min_samples_leaf']],
    #
    #                   max_features_ls[best_results['max_features']],
    #                   np.sqrt(mean_squared_error(data_tr_y, tr_pred)), r2_score(data_tr_y, tr_pred),
    #                   mean_absolute_error(data_tr_y, tr_pred)]
    #     pd_res.append(tr_results)
    #     # validation set
    #     va_pred = best_model.predict(data_va_x)
    #     va_results = [FP_type, split_type, 'va', num_of_compounds,
    #                   n_estimators_ls[best_results['n_estimators']],
    #                   max_depth_ls[best_results['max_depth']],
    #                   min_samples_leaf_ls[best_results['min_samples_leaf']],
    #
    #                   max_features_ls[best_results['max_features']],
    #                   np.sqrt(mean_squared_error(data_va_y, va_pred)), r2_score(data_va_y, va_pred),
    #                   mean_absolute_error(data_va_y, va_pred)]
    #     pd_res.append(va_results)
    #     # test set
    #     te_pred = best_model.predict(data_te_x)
    #     te_results = [FP_type, split_type, 'va', num_of_compounds,
    #                   n_estimators_ls[best_results['n_estimators']],
    #                   max_depth_ls[best_results['max_depth']],
    #                   min_samples_leaf_ls[best_results['min_samples_leaf']],
    #
    #                   max_features_ls[best_results['max_features']],
    #                   np.sqrt(mean_squared_error(data_te_y, te_pred)), r2_score(data_te_y, te_pred),
    #                   mean_absolute_error(data_te_y, te_pred)]
    #     pd_res.append(te_results)
    #     para_res = pd.DataFrame(pd_res, columns=['FP_type', 'split_type', 'type', 'num_of_compounds',
    #                                              'n_estimators', 'max_depth', 'min_samples_leaf',
    #                                               'max_features',
    #                                              'rmse', 'r2', 'mae'])
    n_estimators, max_depth, min_samples_leaf,max_features= n_estimators_ls[best_results['n_estimators']],\
            max_depth_ls[best_results['max_depth']],min_samples_leaf_ls[best_results['min_samples_leaf']],max_features_ls[best_results['max_features']]
    best_results = {'n_estimators':n_estimators, 'max_depth':max_depth, "min_samples_leaf":min_samples_leaf, 'max_features':max_features}
    # print(best_results)
    para_file = str(model_dir).replace('model_save', 'param_save') + '/%s_%s_%s_%s' % (
    split_type, task_type, FP_type, 'RF.param')
    import os
    if not os.path.exists(str(model_dir).replace('model_save', 'param_save')):
        os.makedirs(str(model_dir).replace('model_save', 'param_save'))
    # print(os.path.exists(str(model_dir).replace('model_save', 'param_save')))
    f = open(para_file, 'w')
    f.write('%s' % best_results)
    f.close()
    # pd_res = []
    # # for i in range(10):
    # #     item = best_model_runing((i+1)*500,best_results,X,Y,split_type=split_type,FP_type=FP_type,task_type=task_type,model_dir=model_dir)
    # for i in range(9):
    #     item = best_model_runing((i + 1) , best_results, data, split_type=split_type, FP_type=FP_type,
    #                                  task_type=task_type, model_dir=model_dir)
    #
    #     pd_res.extend(item)
    #
    # if task_type=='cla':
    #     best_res = pd.DataFrame(pd_res, columns=['seed','FP_type', 'split_type', 'type',
    #                                                      'num_of_compounds', 'precision', 'se', 'sp',
    #                                                      'acc', 'mcc', 'auc_prc', 'auc_roc'])
    #
    #     pd1 = para_res[['FP_type', 'split_type', 'type',
    #                                                      'num_of_compounds', 'precision', 'se', 'sp',
    #                                                      'acc', 'mcc', 'auc_prc', 'auc_roc']]
    #     pd1['seed'] =0
    #     best_res = pd.concat([pd1,best_res],ignore_index=True)
    #     best_res = best_res.groupby(['FP_type', 'split_type', 'type'])['acc', 'mcc', 'auc_prc', 'auc_roc'].mean()
    # else:
    #     best_res = pd.DataFrame(pd_res, columns=['seed', 'FP_type', 'split_type', 'type',
    #                                              'num_of_compounds','rmse', 'r2', 'mae'])
    #     pd1 = para_res[['FP_type', 'split_type', 'type',
    #                             'num_of_compounds','rmse', 'r2', 'mae']]
    #     pd1['seed'] = 0
    #     best_res = pd.concat([pd1, best_res], ignore_index=True)
    # return  para_res,best_res

def para_rf(X,Y,args=None,split_type='random',FP_type='ECFP4',task_type = 'cla',model_dir=None):
    param_file = str(model_dir).replace('model_save', 'param_save') + '/%s_%s_%s_%s' % (
    split_type, task_type, FP_type, 'RF.param')
    args = eval(open(param_file, 'r').readline().strip()) if args == None else args
    data = TVT(X, Y)
    data_x, data_te_x, data_y, data_te_y = data.set2ten(0)
    data_tr_x, data_va_x, data_tr_y, data_va_y = split_dataset(data_x, data_y, split_type=split_type, valid_need=False,
                                                               random_state=42, train_size=(8 / 9))
    data_tr_x, data_tr_y = create_des(data_tr_x, data_tr_y, FP_type=FP_type, model_dir=model_dir)
    data_va_x, data_va_y = create_des(data_va_x, data_va_y, FP_type=FP_type, model_dir=model_dir)
    data_te_x, data_te_y = create_des(data_te_x, data_te_y, FP_type=FP_type, model_dir=model_dir)

    pd_res = []
    n_estimators = args['n_estimators']
    max_depth = args['max_depth']
    min_samples_leaf = args['min_samples_leaf']
    max_features = args['max_features']



    best_model = RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features,

                                        n_jobs=6, random_state=1, verbose=0, class_weight='balanced') \
        if task_type == 'cla' else RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,

        n_jobs=6, random_state=1, verbose=0)
    best_model.fit(data_tr_x, data_tr_y)
    if model_dir :
        model_name = str(model_dir) +'/%s_%s_%s_%s'%(split_type,task_type,FP_type,'RF_bestModel.pkl')
        joblib.dump(best_model,model_name)
    num_of_compounds = len(X)
    if task_type == 'cla':
        tr_pred = best_model.predict_proba(data_tr_x)
        tr_results = [FP_type, split_type, 'tr', num_of_compounds,
                      n_estimators,max_depth,min_samples_leaf,max_features]
        tr_results.extend(statistical(data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))
        pd_res.append(tr_results)
        # validation set
        va_pred = best_model.predict_proba(data_va_x)
        va_results = [FP_type, split_type, 'va', num_of_compounds,
                      n_estimators,max_depth,min_samples_leaf,max_features]
        va_results.extend(statistical(data_va_y, np.argmax(va_pred, axis=1), va_pred[:, 1]))
        pd_res.append(va_results)
        # test set
        te_pred = best_model.predict_proba(data_te_x)
        te_results = [FP_type, split_type, 'te', num_of_compounds,
                      n_estimators,max_depth,min_samples_leaf,max_features]
        te_results.extend(statistical(data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
        pd_res.append(te_results)
        para_res = pd.DataFrame(pd_res, columns=['FP_type', 'split_type', 'type', 'num_of_compounds',
                                                 'n_estimators','max_depth','min_samples_leaf','max_features',
                                                 'precision', 'se', 'sp',
                                                 'acc', 'mcc', 'auc_prc', 'auc_roc'])
    else:
        # training set
        tr_pred = best_model.predict(data_tr_x)
        tr_results = [FP_type, split_type, 'tr',  num_of_compounds,
                      n_estimators,max_depth,min_samples_leaf,max_features,
                      np.sqrt(mean_squared_error(data_tr_y, tr_pred)), r2_score(data_tr_y, tr_pred),
                      mean_absolute_error(data_tr_y, tr_pred)]
        pd_res.append(tr_results)
        # validation set
        va_pred = best_model.predict(data_va_x)
        va_results = [FP_type, split_type,'va',  num_of_compounds,
                      n_estimators,max_depth,min_samples_leaf,max_features,
                      np.sqrt(mean_squared_error(data_va_y, va_pred)), r2_score(data_va_y, va_pred),
                      mean_absolute_error(data_va_y, va_pred)]
        pd_res.append(va_results)
        # test set
        te_pred = best_model.predict(data_te_x)
        te_results = [FP_type, split_type, 'te',  num_of_compounds,
                      n_estimators,max_depth,min_samples_leaf,max_features,
                      np.sqrt(mean_squared_error(data_te_y, te_pred)), r2_score(data_te_y, te_pred),
                      mean_absolute_error(data_te_y, te_pred)]
        pd_res.append(te_results)
        para_res = pd.DataFrame(pd_res, columns=['FP_type', 'split_type', 'type', 'num_of_compounds',
                                                 'n_estimators','max_depth','min_samples_leaf','max_features',
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
    import os
    result_dir = model_dir.replace('model_save', 'result_save')
    para_name, best_name = os.path.join(result_dir,
                                        '_'.join([split_type,'RF' , FP_type, 'para.csv'])), os.path.join(
        result_dir, '_'.join([split_type, 'RF', FP_type, 'best.csv']))
    para_res.to_csv(para_name, index=False)
    best_res.to_csv(best_name, index=False)

# df = pd.read_csv('/data/jianping/bokey/OCAICM/dataset/aurorab/aurorab.csv')
# # args = {'n_estimators': 300, 'max_depth': 11, 'min_samples_leaf': 3, 'max_features': 0.8}
# para_rf(df['Smiles'],df['activity'],split_type='scaffold',FP_type='MACCS',task_type = 'cla',model_dir='/data/jianping/bokey/OCAICM/dataset/aurorab/model_save/RF')
# # a,b =tvt_rf(df['Smiles'],df['activity'],split_type='scaffold',FP_type='MACCS',task_type = 'cla',model_dir=False)
# # print(a,b)

