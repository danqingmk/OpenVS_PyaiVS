# Below is a complete example that you can run to test the system:

from script import model_bulid, virtual_screen

# Train and optimize an SVM model using MACCS fingerprints
model_bulid.running('./dataset/abcg2.csv',
                    model=['SVM'],         # Use only SVM
                    FP=['MACCS'],          # Use only MACCS fingerprint
                    split=['random'],      # Use random split
                    run_type='param',      # Optimize hyperparameters
                    cpus=4)                # Use four CPUs

# Retrieve results and model recommendations
model_bulid.running('./dataset/abcg2.csv',
                    model=['SVM'], 
                    FP=['MACCS'],
                    split=['random'],
                    run_type='result')     # Compute and display results

# Screen compounds
virtual_screen.model_screen(model='SVM',
                            split='random',
                            FP='MACCS',
                            model_dir='./abcg2/model_save',
                            screen_file='compounds_to_screen.csv',
                            smiles_col=0)   # First column contains SMILES



