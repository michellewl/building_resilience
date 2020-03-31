# -*- coding: UTF-8 -*-
import os


def create_bjobs():
	j = 0
	for i in range(0, 360, 40):
	    f = open("model" + str(j) + ".bsub", "w+")
	    f.write('''#!/bin/bash \n
	    # BSUB -q short-serial \n
	    # BSUB -J R_job \n
	    # BSUB -oo R-%J.o \n
	    # BSUB -eo R-%J.e \n
	    # BSUB -W 23:00 \n
	    python -c \'import os; os.chdir(\"/home/users/omer/PYTHON/\"); import thresholds as th; th.cube_wrap(-90, 89,''' + str(i) + "," + str(i + 40) + ",\"" + str("HadGEM2-CC") + "\"" +''')\'''')
	    f.close()
	    j += 1


def create_bjobs_era():
	j = 0
	for i in range(0, 360, 40):
	    f = open("era" + str(j) + ".bsub", "w+")
	    f.write('''#!/bin/bash \n
	    # BSUB -q short-serial \n
	    # BSUB -J R_job \n
	    # BSUB -oo R-%J.o \n
	    # BSUB -eo R-%J.e \n
	    # BSUB -W 23:00 \n
	    python -c \'import os; os.chdir(\"/home/users/omer/PYTHON/\"); import thresholds as th; th.get_threshold_world(-90, 89,''' + str(i) + "," + str(i + 40) +  ''')\'''')
	    f.close()
	    j += 1