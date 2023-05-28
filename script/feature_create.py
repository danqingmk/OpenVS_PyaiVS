import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
np.random.seed(43)
def create_des(X,Y ,FP_type = 'ECFP4',model_dir=False):
    smiles = np.array(X)
    if FP_type == 'ECFP4':
        ms = [AllChem.MolFromSmiles(smiles[i]) for i in range(len(smiles))]
        ecfpMat = np.zeros((len(ms), 1024), dtype=int)
        for i in range(len(ms)):
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(ms[i], 2, 1024)
                ecfpMat[i] = np.array(list(fp.ToBitString()))
            except:
                ecfpMat[i] = np.array([0]*1024)
        X = ecfpMat
    elif FP_type == 'MACCS':
        ms = [AllChem.MolFromSmiles(smiles[i]) for i in range(len(smiles))]
        ecfpMat = np.zeros((len(ms), 167), dtype=int)
        for i in range(len(ms)):
            fp = AllChem.GetMACCSKeysFingerprint(ms[i])
            ecfpMat[i] = np.array(list(fp.ToBitString()))
        X = ecfpMat
    elif FP_type == '2d-3d':
        dataset = str(model_dir).split('/model_save')[0].split('/')[-1]
        des_file = str(model_dir).split('/model_save')[0] +'/'+dataset+'_23d_adj.csv'
        des = pd.read_csv(des_file).drop_duplicates()
        des = des.set_index('Name')
        des = des.dropna(axis=1)
        features = len(des.columns)
        ecfpMat = np.zeros((len(X), features), dtype=float)
        for j,smile in enumerate(smiles):
            index = str(smile)
            try:
                ecfpMat[j] = np.array(des.loc[index,:])
            except:
                pass
        X = ecfpMat
    elif FP_type == 'pubchem':
        dataset = str(model_dir).split('/model_save')[0].split('/')[-1]
        des_file = str(model_dir).split('/model_save')[0] + '/' + dataset + '_pubchem.csv'
        des = pd.read_csv(des_file).drop_duplicates()
        des = des.set_index('Name')
        features = len(des.columns)
        ecfpMat = np.zeros((len(X), features), dtype=int)
        for j, smile in enumerate(smiles):
            index = str(smile)
            try:
                ecfpMat[j] = np.array(des.loc[index, :])
            except:
                pass
        X = ecfpMat
    return X,Y