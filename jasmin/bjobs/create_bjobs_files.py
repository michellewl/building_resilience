# -*- coding: UTF-8 -*-
import os


def create_bjobs():
	j = 0
	for i in range(0, 356, 5):
		f = open("model" + str(j) + ".bsub", "w+")
	    f.write('''#!/bin/bash \n
	    # BSUB -q short-serial \n
	    # BSUB -J R_job \n
	    # BSUB -oo R-%J.o \n
	    # BSUB -eo R-%J.e \n
	    # BSUB -W 23:00 \n
	    python -c \'import os; os.chdir(\"/home/users/omer/PYTHON/\"); import threshold as th; th.model_wrap(-90, 89,''' + str(i) + "," + str(i + 5) + ''')\'''')
	    f.close()
	    j += 1


