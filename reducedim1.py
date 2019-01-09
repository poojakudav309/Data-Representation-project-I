# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:20:50 2018

@author: pooja
"""

import numpy as  np
import sys


if len(sys.argv) != 5 :
  print(sys.argv[0], "takes 4 arguments. Not ", len(sys.argv)-1)
  sys.exit()

first = sys.argv[1]
second = sys.argv[2]
third =sys.argv[3]
four =sys.argv[4]

Xt=np.genfromtxt(first,delimiter=',')
y=np.genfromtxt(second,autostrip=True)
R=np.dot(Xt.T,Xt)
evals,evecs=np.linalg.eigh(R)
idx=np.argsort(evals)[::-1]
evals=evals[idx]
evecs=evecs[:,idx]
r=2
V_r=evecs[:,:r]
result=np.dot(Xt,V_r)


np.savetxt(third, V_r.T,delimiter=',')
np.savetxt(four, result,delimiter=',')