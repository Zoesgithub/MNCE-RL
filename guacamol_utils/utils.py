from loguru import logger
import guacamol
from guacamol.benchmark_suites import goal_directed_benchmark_suite
from collections import OrderedDict
import json
from rdkit import Chem
import time
import numpy as np
import os
from guacamol.utils.data import get_time_string




def ana_generated_molecules_with_benchmark(Molecules, benchmark):
    logger.info("Running evaluations on {}".format(benchmark.name) )
    results=benchmark.ana_molecules(Molecules, time.time())
    logger.info("Results for the benchmark {}".format(results.benchmark_name))
    logger.info("   Score:{}".format(results.score))
    logger.info("   Metadata:{}".format(results.metadata))
    return results

def make_initpop(path, benchmark, topk=512):
    if os.path.isfile(path):
        with open(path, "r") as f:
            content=f.readlines()
            if len(content)==topk:
                return content
    with open("Data/guacamol/train.txt", "r") as f:
        content=f.readlines()
    scores=[benchmark.ana_molecules([x], time.time()).score for x in content]
    content=list(zip(content, scores))
    content=sorted(content, key=lambda x: x[1])
    content=content[-topk:]
    with open(path, "w") as f:
        for l in content:
            f.writelines(l[0])
    return [x[0] for x in content]

def ana_generated_molecules(Molecules, benchmark_version="v2", save_path=None):
    benchmarks=goal_directed_benchmark_suite(version_name=benchmark_version)
    assert len(Molecules)==len(benchmarks)
    results=[]
    for molecules, benchmark in zip(Molecules, benchmarks):
        results.append(ana_generated_molecules_with_benchmark(molecules, benchmark))
    benchmark_results=OrderedDict()
    benchmark_results['guacamol_version'] = guacamol.__version__
    benchmark_results['benchmark_suite_version'] = benchmark_version
    benchmark_results['timestamp'] = get_time_string()
    benchmark_results['results'] = [vars(result) for result in results]

    if save_path:
        with open(save_path, "wt") as f:
            f.write(json.dumps(benchmark_results, indent=4))
    return benchmark_results

class benchmark_funcwrapper(object):
    def __init__(self, func, weight=1.0, moving_rate=1.0):
        self.func=func
        self.weight=weight
        self.moving_rate=moving_rate
        self.molecule=[None, -1.0]
        self.mean=0.0
        self.var=0.0
        self.n=0
        self.smilelist=[]
        self.smiledict={}


    def __call__(self, mol):
        smile=Chem.MolToSmiles(mol)
        score=self.func.ana_molecules([smile], time.time())
        ret=score.score#*self.weight
        self.mean=self.mean*0.999+score.score*0.001
        self.var=self.var*0.99+(score.score-self.mean)**2*0.01
        self.n+=1

        if self.n%3200==0:
            print(self.mean, self.var)
            logger.info("The best molecules: {}".format(self.smilelist))

        if score.score>self.molecule[1]:
            self.molecule=[smile, score.score]
            print(smile, score.score, self.mean)

        if len(self.smilelist)<300:
            i=0
            if not smile in self.smiledict:
                while i<len(self.smilelist) and self.smilelist[i][1]<score.score:
                    i+=1
                    continue
                self.smilelist.insert(i, [smile,score.score])
                self.smiledict[smile]=1.0
        elif score.score>self.smilelist[0][1]:
            if not smile in self.smiledict:
                i=0
                while i<len(self.smilelist) and self.smilelist[i][1]<score.score:
                    i+=1
                    continue
                self.smilelist.insert(i, [smile, score.score])
                self.smiledict.pop(self.smilelist[0][0])
                self.smilelist.pop(0)
                self.smiledict[smile]=1.0
        ret=ret*self.weight#+(np.random.rand()-0.5)*0.1
        return score.score, ret


    def score(self, mol):
        return self.__call__(mol)[0]
