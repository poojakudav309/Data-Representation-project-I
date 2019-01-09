# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:54:50 2018

@author: pooja
"""
import numpy as  np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages   


if len(sys.argv) != 5 :
  print(sys.argv[0], "takes 4 arguments. Not ", len(sys.argv)-1)
  sys.exit()

first = sys.argv[1]
second = sys.argv[2]
third =sys.argv[3]
four =sys.argv[4]


Xt=np.genfromtxt(first,delimiter=',')
y=np.genfromtxt(second,autostrip=True)

mu=np.mean(Xt.T,axis=1)

Xc=Xt-mu[None,:]
C=np.dot(Xc.T,Xc)
evals,evecs=np.linalg.eigh(C)

idx=np.argsort(evals)[::-1]
evals=evals[idx]
evecs=evecs[:,idx]
r=2
V_r=evecs[:,:r]
result=np.dot(Xt,V_r)



x1 = result[:, 0] # first column of X
x2 = result[:, 1] # second column of X



fig, ax = plt.subplots()
    
ax.scatter(x1, x2)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title("PCA with mean subrtraction")

# save to pdf
plotname = 'pca_2_plot'
pdf = PdfPages(plotname + '.pdf')
pdf.savefig(fig)
pdf.close()

np.savetxt(third, V_r.T,delimiter=',')
np.savetxt(four, result,delimiter=',')