import argparse
from rdkit import Chem
from loguru import logger
from rdkit.Chem import AllChem
from rdkit import DataStructs
from mol_distance import mol_diversity

parser=argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="The path to the file")
args=parser.parse_args()

File=args.file
iradius=4
nBits=2048
useChirality=True
with open(File, "r") as f:
    content=f.readlines()
divisity=mol_diversity(content)
logger.info("The diversity for {} is {}".format(File,divisity))

"""SMILES=[chem.MolFromSmiles(x) for x in content]
FPS=[]

for i,x in enumerate(SMILES):
    if i%1000==0:
        logger.info("Computing Morgan FP to {}".format(i))
    FPS.append(AllChem.GetMorganFingerprint(x,radius, nBits=nBits, useChirality=useChirality))


return DataStructs.TanimotoSimilarity(x, target) """
