import os
import shutil
from PyaiVS import compute_info
from PyaiVS import data_utils
from PyaiVS import run_model,run_result
os.environ['PYTHONHASHSEED'] = str(42)  # 为了禁止hash随机化，使得实验可复现
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import time



def running(file_name,out_dir = './',split=None,model=None,FP=None, run_type=None,cpus=4):
    split = split if type(split) != type(None) else ['random']
    model = model if type(model) != type(None) else ['SVM']
    FP = FP if type(FP) != type(None) else ['MACCS']
    FP_list = ['2d-3d','MACCS','ECFP4','pubchem','all']
    split_list = ['random', 'scaffold', 'cluster','all']
    model_list = ['SVM','KNN','DNN','RF','XGB','gcn','gat','attentivefp','mpnn','all']
    split = ['random', 'scaffold', 'cluster'] if 'all' == split else split
    FP =['MACCS','ECFP4','pubchem'] if 'all' == FP else FP
    model = ['SVM','KNN','DNN','RF','XGB','gcn','gat','attentivefp'] if 'all' == model else model
    assert len(set(split) - set(split_list)) == 0, '{} element need in {}'.format(split, split_list)
    assert len(set(model)-set(model_list))==0,'{} element need in {}'.format(model ,model_list)
    assert len(set(FP) - set(FP_list)) == 0, '{} element need in {}'.format(FP, FP_list)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    assert len(set(list([file_name.split('.')[-1]])) -set(list(['csv','txt','tsv'])))==0, '{} need in ["csv","txt","tsv"]'.format(file_name.split('.')[-1])
    dataset = file_name.split('/')[-1].split('.'+file_name.split('.')[-1])[0]
    dir_path = os.path.join(out_dir, dataset)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file = os.path.join(dir_path, dataset + '.csv')
    if not file_name==os.path.join(dir_path,dataset + '.csv'):
        shutil.copy(file_name, os.path.join(dir_path,dataset + '.csv'))
    des = [fp for fp in FP if fp in ['pubchem','2d-3d']]
    data_utils.start(file, des=des)
    mpl = True if cpus>1 else False
    if run_type =='param':
        run_model.main(file, split, FP, model, cpus, mpl, 'cpu')
    else:
        run_result.main(file, split, FP, model, cpus, mpl, 'cpu')
        compute_info.get_info(file_name=file, models=model, split=split, FP=FP)
        compute_info.para(file_name=file)

if __name__ == '__main__':
    running('/data/jianping/bokey/OCAICM/dataset/aurorab/aurorab.csv',
            # out_dir='/data/jianping/bokey/OCAICM/dataset/',split='all',model=['XGB'],FP=['ECFP4'], cpus=4,run_type='t')#,'DNN'
            out_dir='/data/jianping/bokey/OCAICM/dataset/', split=['scaffold'], model=['gcn'], FP=['MACCS'], cpus=0,run_type='param'
                                                                                                                             )
