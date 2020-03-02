import scipy.io
import numpy as np

def readmat():

 infile='/home/binder/entwurf6/codes/tfplay/ai/flowers_data/imagelabels.mat'
 outfile='/home/binder/entwurf6/codes/tfplay/ai/flowers_data/imagelabels.npy'
 labelsdict=scipy.io.loadmat(infile)
 
 print(type(labelsdict))
 
 for keyval in labelsdict.items():
   print(keyval[0],keyval[1])
   
 lb=labelsdict['labels']
 
 labels=np.squeeze(lb-1)
 
 
 
 print(np.min(labels),labels.shape, lb.shape)
 
 np.save(outfile,labels)
 
if __name__=='__main__':
  readmat()
