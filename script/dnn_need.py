from PyaiVS import model_bulid,virtual_screen
model_bulid.running('/data/jianping/bokey/test/abcg2.csv',
                    out_dir='/data/jianping/bokey/test/dataset',
                    split='all',
                    model='all',
                    FP='all',
                    run_type='param',
                    cpus=4)
model_bulid.running('/data/jianping/bokey/test/abcg2.csv',
                    out_dir='/data/jianping/bokey/test/dataset',
                    split='all',
                    model='all',
                    FP='all',
                    run_type='result',
                    cpus=4)
# virtual_screen.model_screen(model='XGB',
#                             split='random',
#                             FP='ECFP4',
#                             prop=0.8,
#                             model_dir='/data/jianping/bokey/test/dataset/abcg2/model_save',
#                             screen_file='/data/jianping/bokey/OCAICM/dataset/a1/base.csv',
#                             sep=';',
#                             smiles_col='smiles')

