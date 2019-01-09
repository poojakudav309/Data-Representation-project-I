# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:20:29 2018

@author: pooja
"""
import sys
import numpy as np
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

m,n=Xt.shape

unique_labels =set(y)


W_class=np.zeros((n,n))
B_class=np.zeros((n,n))
for c in unique_labels:
    s1 = np.zeros((1,n))
    C1 =np.zeros((n,n))
    m1 = 0
    for i, xt in enumerate(Xt) :
        if y[i] ==c :
            s1 += xt
            m1 += 1
     
    mu1 = s1/m1
    
    for i, xt in enumerate(Xt) :
        if y[i] == c:
            ct=xt-mu1
            C1 +=np.dot(ct.T,ct)
    W_class += C1

""""    
    
print("Within Class Scatter:")    
print(W_class)

"""

evals,evecs=np.linalg.eigh(W_class)
idx=np.argsort(evals)[::-1]
evecs=evecs[:,idx]
evals=evals[idx]

r=1
V_r=evecs[:,:r]


r=2
V_r=evecs[:,:r]
result=np.dot(Xt,V_r)
  


x1 = result[:, 0] # first column of X
x2 = result[:, 1] # second column of X



fig, ax = plt.subplots()
ax.scatter(x1, x2)
plt.title("Minimize within class scatter")
plt.xlabel("PC1")
plt.ylabel("PC2")

# save to pdf
plotname = 'w_class_scatter_plot'
pdf = PdfPages(plotname + '.pdf')
pdf.savefig(fig)
pdf.close()

np.savetxt(third, V_r.T,delimiter=',')
np.savetxt(four, result,delimiter=',')
