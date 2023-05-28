from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem import Draw


mol = Chem.MolFromSmiles('CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C')
scaffold = MurckoScaffoldSmiles(mol=mol)
mol2 = Chem.MolFromSmiles(scaffold)

img=Draw.MolToImage(mol)
img.save('picture.png')

img=Draw.MolToImage(mol2)
img.save('picture2.png')
