from script import model_bulid,virtual_screen
model_bulid.running('./dataset/abcg2.csv',
                    out_dir='./dataset',
                    run_type='param',
                    cpus=4)
model_bulid.running('./dataset/abcg2.csv',
                    out_dir='./dataset',
                    run_type='result',
                    cpus=4)
virtual_screen.model_screen(model='SVM',
                            split='random',
                            FP='MACCS',
                            model_dir='./dataset/abcg2/model_save',
                            screen_file='./database/base.csv',
                            sep=';',
                            smiles_col='smiles')


