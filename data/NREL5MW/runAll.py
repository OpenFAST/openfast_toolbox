import os
from openfast_toolbox.case_generation.runner import *


if __name__=='__main__':
    EXE = './openfast.exe'
    files = ['./Main_Onshore.fst', './Main_Onshore_OF2_BD.fst', '5MW_Land_Lin_BladeOnly/Main.fst', '5MW_Land_Lin_Rotating/Main.fst']
    run_cmds(files, EXE, parallel=True, showOutputs=True, nCores=4, showCommand=True)

    #for f in files:
    #    parentDir = os.path.dirname(f)
    #    removeFASTOuputs(f)

