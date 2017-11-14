import glob
import re
import json
import pickle
import gzip
from collections import Counter
import plyvel
import concurrent.futures
import sys
import os
import numpy as np

HOME = os.environ['HOME']
class Data:
  def __init__(self):
    self.d = {}
    self.data_owner_ids = set()

def _map1(name):
  try:
    number = name.split('-').pop()
    db = plyvel.DB('step1/{number}.ldb'.format(number=number), create_if_missing=True)
    f = open(name)
    for index, line in enumerate(f):
      if index%1000 == 0:
        print('now iter', index, '/', number)
      line = line.strip()
      tuuid, obj = line.split('\t')
      tuuid = tuuid.split('_')[0]
      if db.get( bytes(tuuid, 'utf8') ) is not None:
        continue
      obj = json.loads( obj )
      all_keys = sum([keys for time, keys in obj.items()], [] )

      d = dict(Counter(all_keys))
      data_owner_ids = set( [ doi.split('_').pop() for doi in d.keys() ] )
      data = Data()
      data.d = d
      data.data_owner_ids = data_owner_ids
      db.put( bytes(tuuid, 'utf8'), gzip.compress(pickle.dumps(data) ) )
  except Exception as e:
    print('Some Deep Error', e)

if '--map1' in sys.argv:
  names = [name for name in glob.glob(f'{HOME}/sda/part-*')]
  print( names )
  with concurrent.futures.ProcessPoolExecutor(max_workers=16) as exe:
    exe.map( _map1, names )

if '--map2' in sys.argv:
  keyword_freq = {}
  for name in glob.glob('step1/*'):
    print(name)
    db = plyvel.DB(name, create_if_missing=False)
    for index, (key, val) in enumerate(db):
      if index%1000 == 0:
        print('now iter', index)
      key = key.decode('utf8')
      data = pickle.loads( gzip.decompress(val) )
      for keyword in data.d.keys():
        if keyword_freq.get(keyword) is None:
          keyword_freq[keyword] = 0
        keyword_freq[keyword] += 1
    open('keyword_freq.json', 'w').write( json.dumps(keyword_freq, indent=2, ensure_ascii=False) )    

class A:
  def __init__(self):
    self.norm = None
    self.data_owner_ids = None

if '--filter1' in sys.argv:
  keyword_freq = json.loads(open('keyword_freq.json').read() ) 
  keyword_index = {}

  f1db = plyvel.DB('filter1.ldb', create_if_missing=True)
  for keyword, freq in sorted(keyword_freq.items(), key=lambda x:x[1]*-1)[:10000] :
    keyword_index[keyword] = len( keyword_index )
  for name in glob.glob('step1/*'):
    db = plyvel.DB(name, create_if_missing=False)
    for index, (bkey, val) in enumerate(db):
      if index%1000 == 0:
        print('now iter', index)
      key = bkey.decode('utf8')
      # print(key)
      data = pickle.loads( gzip.decompress(val) )
      if len(data.data_owner_ids) >= 2:
        base = [0.0]*len(keyword_index)
        for key, freq in data.d.items():
          if keyword_index.get(key) is None:
            continue
          base[ keyword_index[key] ]  = freq

        su   = sum(base)
        if su == 0.0:
          continue
        base = [b/su for b in base]
        base = np.array(base)
        a = A()
        a.norm = base
        a.data_owner_ids = data.data_owner_ids
        f1db.put(bkey, gzip.compress( pickle.dumps(a) ) )

if '--make_npy' in sys.argv:
  f1db = plyvel.DB('filter1.ldb', create_if_missing=False)
  norms = []
  for index, (bkey, val) in enumerate(f1db):
    val = pickle.loads( gzip.decompress(val) )
    #print( val.norm.tolist() )
    #if np.isfinite(val.norm).any() :
    # continue
    print(bkey.decode('utf8'))
    norms.append( val.norm )
    if index > 1000000:
      break
  norms = np.array( norms )
  np.save('norms', norms)
  print( norms.shape )
