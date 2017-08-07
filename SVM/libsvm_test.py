# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:45:56 2017

@author: bowen
"""

from svmutil import *
from svm import *
import time
    
#y, x = [1,-1], [{1:1, 2:1}, {1:-1,2:-1}]
y,x = svm_read_problem('ijcnn1_train.txt')
yt,xt = svm_read_problem('ijcnn1_test.txt')
prob  = svm_problem(y, x)
#2 0 1 3 4
param = svm_parameter('-t 2 -c 500 -b 1')
start = time.clock()
model = svm_train(prob, param)
#yt = [1]
#xt = [{1:1, 2:1}]
p_label, p_acc, p_val = svm_predict(yt, xt, model)
end = time.clock()
#svm_save_model('model_file', model)
#print(p_label)
print ('running time :')
print (end-start)