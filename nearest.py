# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:15:10 2018

@author: pooja
"""


import numpy as np
import sys



if len(sys.argv) != 6 :
  print(sys.argv[0], "takes 2 arguments. Not ", len(sys.argv)-1)
  sys.exit()

first = sys.argv[1]
second = sys.argv[2]
third =sys.argv[3]
four =sys.argv[4]
five = sys.argv[5]


result=np.genfromtxt(first,delimiter=',')

y=np.genfromtxt(third,autostrip=True)

V_r=np.genfromtxt(second,delimiter=',')

qr=np.genfromtxt(four,delimiter=',')

train_result=np.dot(qr,V_r.T)

re=[]
label=np.genfromtxt(five)
cs=list(label)

if int(-1) in cs: 
    for tr in train_result:
        l=[]
        for i,res in enumerate(result):
            dist = np.linalg.norm(res-tr)
            l.append(dist)
        re.append(int(l.index(min(l))))
else:
    re=[]

    for tr,c in zip(train_result,cs):
        l=[]
        indx=[]
        for i,res in enumerate(result):
            if y[i]==c:
                dist = np.linalg.norm(res-tr)
                l.append(dist)
                indx.append(i)     
        re.append(int(indx[int(l.index(min(l)))]))


# read the files for the matrices reduced_Xt, vector, labels, queried_point and label..



# return index of nearest neighbor 
res = [int(x) for x in re]
nn_idx = np.asarray(res)



# Output file name should be read from command line.
output_file = 'nn_idx.csv' 

# save output in comma separated filename.txt.
np.savetxt(output_file, nn_idx, delimiter=',')

with open(output_file,"w") as f:
    for line in res:
        f.write(str(line) + "\n")
        
