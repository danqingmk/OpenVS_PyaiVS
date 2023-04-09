import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski, Descriptors, Crippen
import multiprocessing as mp
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import argparse
import joblib
import time
import os

s = time.time()
params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog = FilterCatalog(params)


def ro5(x):
    try:
        mol = Chem.MolFromSmiles(x)
        h_acc = Lipinski.NumHAcceptors(mol)
        h_don = Lipinski.NumHDonors(mol)
        # rotal = Lipinski.NumRotatableBonds(mol)
        weight = Descriptors.ExactMolWt(mol)
        logp = Crippen.MolLogP(mol)
        count = sum([weight <= 500, h_acc <= 10, h_don <= 5, logp <= 5])
        return 1 if count >= 3 else 0
    except:
        return 0


def pains(x):
    mol = Chem.MolFromSmiles(x)
    entry = catalog.GetFirstMatch(mol)  # Get the first matching PAINS
    return 1 if entry is not None else 0


def fp(X, type='ECFP4', file=''):
    smiles = np.array(X)
    count = 0
    if type == 'ECFP4':
        ms = [Chem.MolFromSmiles(smiles[i]) for i in range(len(smiles))]
        ecfpMat = np.zeros((len(ms), 1024), dtype=int)
        for i in range(len(ms)):
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(ms[i], 2, 1024)
                ecfpMat[i] = np.array(list(fp.ToBitString()))
            except:
                count += 1
                ecfpMat[i] = np.zeros((1, 1024), dtype=int)
        X = ecfpMat

        return X
    elif type == 'MACCS':
        ms = [Chem.MolFromSmiles(smiles[i]) for i in range(len(smiles))]
        ecfpMat = np.zeros((len(ms), 167), dtype=int)
        for i in range(len(ms)):
            fp = AllChem.GetMACCSKeysFingerprint(ms[i])
            ecfpMat[i] = np.array(list(fp.ToBitString()))
        X = ecfpMat
        return X
    elif type == 'pubchem':
        des = pd.read_csv(file)
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
        return X


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='we must give this para')
    parser.add_argument('--models', required=True, nargs='*')
    parser.add_argument('--prop', default=0.5, type=float)
    parser.add_argument('--sep', default=',', type=str)
    parser.add_argument('--smiles_col', default='smiles', type=str)
    parser.add_argument('--cpus', default=1, type=int)
    parser.add_argument('--out_dir')
    args = parser.parse_args()
    return args


def cir_file(file=None, sep=',', models=None, prop=0.5, out_dir='./', smiles_col='Smiles'):
    count = 0
    com = 0
    out_file = os.path.join(out_dir, file.split('/')[-1].replace('.csv', '_screen_{}.csv'.format(prop)))
    f = open(out_file, 'w')
    print()
    assert smiles_col in open(file, 'r').readline().strip().split(sep), '{} is not in screen_file columns'.format(
        smiles_col)
    local = open(file, 'r').readline().strip().split(sep).index(smiles_col)

    for line in open(file, 'r'):

        if line.startswith('smiles') or line.startswith('Smiles'):
            continue
        com += 1
        smiles = line.strip().split(sep)[local]
        model = models

        type = model.split('/')[-1].split('_')[2]
        fprint = fp([smiles], type=type, file=file)

        model = joblib.load(model)
        pred = model.predict_proba(fprint)

        if pred[0][1] >= prop:
            lpsk = ro5(smiles)
            if lpsk == 1:
                count += 1
                f.write('{}\n'.format(smiles))
        if count / 10000 == 0:
            # print(count)
            pass
    f.close()
    print('screen precent ', round((count / com), 2))


if __name__ == '__main__':
    args = parse_args()
    file = args.file
    models = args.models
    sep = args.sep
    prop = args.prop
    smiles_col = args.smiles_col
    cpus = args.cpus
    out_dir = args.out_dir

    if os.path.isdir(file):
        p = mp.Pool(processes=cpus)
        for file_content in os.listdir(file):
            if '.csv' in file_content and 'pubchem' not in file_content:
                file_path = os.path.join(file, file_content)
                param = {'file': file_path, 'sep': sep, 'models': models, 'prop': prop, 'out_dir': out_dir}
                get = p.apply_async(cir_file, kwds=param)
        p.close()
        p.join()
    elif os.path.isfile(file):
        cir_file(file=file, sep=sep, models=models, prop=prop, out_dir=out_dir)
    else:
        print('What\'s this ?')
