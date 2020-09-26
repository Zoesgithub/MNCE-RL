import argparse
from rdkit.Chem import Draw
from rdkit import Chem
import sys
sys.path.append("..")
from utils.datautils import reward_penalized_log_p
from rdkit.Chem.Descriptors import qed
import cairosvg

parser=argparse.ArgumentParser()
parser.add_argument("-f", "--file")
args=parser.parse_args()
with open(args.file, "r") as f:
    content=f.readlines()

content=content[-20:]
content=[Chem.MolFromSmiles(x) for x in content][::-1]
logps=[reward_penalized_log_p(x) for x in content]
qeds=[qed(x) for x in content]
#legend=["QED={}".format(round(x,3)) for x in qeds]
if "logp" in args.file:
    legend=["Penalized logp={}".format(round(x,2)) for x in logps]
elif "qed" in args.file:
    legend=["QED={}".format(round(x,3)) for x in qeds]
else:
    legend=None
assert len(legend)==len(content)
print(logps[-3:])
print(qeds[-3:])
string=Draw.MolsToGridImage(content,molsPerRow=4, subImgSize=(250,250), legends=legend, useSVG=True)#.save("{}pic.png".format(args.file.split("\.")[0].split("/")[1]))
cairosvg.svg2png(bytestring =string, dpi=10000, write_to="{}pic.png".format(args.file.split("\.")[0].split("/")[1]))
