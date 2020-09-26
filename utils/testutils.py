import torch
import torch.nn as nn
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")
def ismole(smile):
    m=Chem.MolFromSmiles(smile)
    if m is None:
        return False
    return True
