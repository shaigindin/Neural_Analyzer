import fnmatch
import os

DIR = "/home/mercydude/Desktop/Neural_Analyzer/files/out/EYES/PURSUIT_FRAGMENTS/"
all_cell_names = fnmatch.filter(os.listdir(DIR), '*')

for name in all_cell_names:
    l = name.split('_')
    newName = l[0] + l[1]
    os.rename(DIR + name, DIR + newName.upper())
