import numpy as np

import pickle

import gzip

from sklearn.cluster import KMeans

import sys

if '--fit' in sys.argv:
  norms = np.load('norms.npy')
  print( norms )
  model = KMeans(n_clusters=300, random_state=0, verbose=1, n_jobs=-1, max_iter=100)
  print( 'start to fit' )
  model.fit(norms)
  print( 'end to fit' )
  open('model.pkl','wb').write( gzip.compress( pickle.dumps(model) ) )

if '--center' in sys.argv:
  model = pickle.loads( gzip.decompress( open('model.pkl','rb').read() ) )
  cs = model.cluster_centers_.tolist()
  for c in cs:
    print(c)

