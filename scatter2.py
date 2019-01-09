# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:20:39 2018

@author: pooja
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  
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
m,n=Xt.shape
unique_labels =set(y)


    
    
mu_global=np.mean(Xt,axis=0)

W_class=np.zeros((n,n))
B_class=np.zeros((n,n))
for c in unique_labels:
    B1 = np.zeros((n,n))
    BT1 = np.zeros((n,n))
    s1 = np.zeros((1,n))
    C1 =np.zeros((n,n))
    #B_cls=np.zeros((4,4))
    m1 = 0
    for i, xt in enumerate(Xt) :
        if y[i] ==c :
            B1 += np.dot(xt.T,xt)
            s1 += xt
            m1 += 1
    mu1 = s1/m1
    
    for i, xt in enumerate(Xt) :
        if y[i] == c:
            ct=xt-mu1
            bt=mu1-mu_global
            C1 +=np.dot(ct.T,ct)
            BT1+=np.dot(bt.T,bt)
    B_class += BT1
    

evals,evecs=np.linalg.eigh(B_class)
idx=np.argsort(evals)[::-1]

evecs=evecs[:,idx]
evals=evals[idx]

r=2
V_r=evecs[:,:r]

result=np.dot(Xt,V_r)





x1 = result[:, 0] # first column of X
x2 = result[:, 1] # second column of X



fig, ax = plt.subplots()
ax.scatter(x1, x2)
plt.title("Maximize between class scatter")
plt.xlabel("PC1")
plt.ylabel("PC2")


# save to pdf
plotname = 'scatter_2_plot'
pdf = PdfPages(plotname + '.pdf')
pdf.savefig(fig)
pdf.close()

np.savetxt(third, V_r.T,delimiter=',')
np.savetxt(four, result,delimiter=',')

