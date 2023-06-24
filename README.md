# What is PyaiVS ?
The tool can complete the construction of diffierent dataset classification models with only one line of code, and recommend the optimal model for virtual screening.  
The tool integrates multiple machine learning models, common molecular descriptors and three data set splitting methods.   
The tool integrates the virtual screening function and uses the optimal model to screen the quick screening of the compound library provided by users through a single code.



# How to use this tool ?

### 1 Bulid the environment

1. conda create -n envir_name python=3.8              
2. conda install rdkit rdkit             
3. conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch  # need to confirm your cuda>=10.2  
4. conda install -c dglteam dgl==0.4.3post2   
5. conda install **xgboost,hyperopt,mxnet,requests,mdtraj**    
6. pip install PyaiVS   


### 2 Parameter setting

    >>>from PyaiVS import model_bulid
    >>>model_bulid.running('abcg2.csv',model=['SVM','DNN'],run_type='param')
    >>>model_bulid.running('abcg2.csv',model=['SVM','DNN'],run_type='result')
           model    des   split   auc_roc  f1_score       acc       mcc
         2   SVM  ECFP4  random  0.969047  0.903497  0.917723  0.831872
         4   DNN  ECFP4  random  0.961781  0.881708  0.898430  0.426201
    (The result output is the optimal model recommendation order considering mcc)

### 3 Generated Files



    >>>from PyaiVS import virtul_screen
    >>>virtul_screen.model_screen(model='attentivefp',FP= None,split='random',screen_file='/tmp/screen',prop = 0.5,sep = ',',
        model_dir='./dataset/abcg2/model_save/',smiles_col=0)
    (Finally, folder screen will be generated under the set model_save peer directory to store the filtering results)

## 4 Us Exmample : the ABCG2 inihbits model buliding

* 1 $ python example.py
* 2 Obtain multiple model calculation results and MCC ranking, and store the specific data content in ./dataset/abcg2
