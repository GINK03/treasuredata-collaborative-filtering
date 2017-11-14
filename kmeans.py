import numpy as np

import pickle

import gzip

from sklearn.cluster import KMeans

import sys

if '--fit' in sys.argv:
  norms = np.load('norms.npy')
  print( norms )
  model = KMeans(n_clusters=1000, random_state=0)
  print( 'start to fit' )
  model.fit(norms)
  open('model.pkl','wb').write( gzip.compress( pickle.dumps(model) ) )

