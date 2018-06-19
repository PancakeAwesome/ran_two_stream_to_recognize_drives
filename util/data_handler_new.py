import sys
import h5py
import numpy as np
import time
import os
import math

videoDir = '/home/jie/UCF101/new'

class DataHandler(object):

  def __init__(self, data_pb):
    self.seq_length_ = data_pb.seq_length		# no of timesteps
    self.seq_stride_ = data_pb.stride			# stride for overlap
    self.randomize_ = data_pb.randomize			# randomize their order for training
    self.batch_size_ = data_pb.batch_size		# batch size
    self.row_ = 0
    
    labels_ = self.GetLabels(data_pb.labels_file)	# labels
    self.labels_ = np.array(labels_, dtype=np.int64)
  
    f5 = h5py.File(data_pb.data_file,'r')
    self.handler = f5[data_pb.dataset_name]
    self.indices_ = range(self.handler.shape[0])
    assert self.handler.shape[0] == self.labels_.shape[0]

    #print 'data frames number: %d' % data.shape[0]
    self.frame_size_ =  1024   #data.shape[1:4]					# 3D cube  [1024,7,7]
    self.dataset_name_ = data_pb.dataset_name

    self.dataset_size_ = self.handler.shape[0]
    print 'Dataset size', self.dataset_size_

    self.Reset()
    
    '''
    self.batch_data_  = np.zeros((self.seq_length_, self.batch_size_, self.frame_size_, 7, 7), dtype=np.float32)
    if data_pb.dataset != 'h2mAP':
      self.batch_label_ = np.zeros((self.seq_length_, self.batch_size_), dtype=np.int64)
    else:
      self.batch_label_ = np.zeros((self.seq_length_, self.batch_size_, 12), dtype=np.int64)
    '''

  def GetBatch(self, data_pb, verbose=False):
    
    #self.batch_data_  = np.zeros((self.seq_length_, self.batch_size_, self.frame_size_, 7, 7), dtype=np.float32)
  
    endi = min([self.row_+ self.batch_size_, self.dataset_size_])
    idx = self.indices_[self.row_:endi]
    idx = np.sort(idx)
    n_examples = len(idx)
    self.batch_data_ = self.handler[idx,:,:,:,:]
    self.batch_label_ = np.tile(self.labels_[idx].reshape((1,n_examples)),(self.seq_length_,1))
    self.row_ = endi
    if n_examples < self.batch_size_:
      self.batch_data_ = np.concatenate((self.batch_data_, np.zeros((self.batch_size_-n_examples, self.seq_length_, self.frame_size_, 7, 7))), axis=0)
      self.batch_label_ = np.concatenate((self.batch_label_, np.zeros((self.seq_length_, self.batch_size_-n_examples))), axis=1)
    if self.row_ ==  self.dataset_size_:
      self.Reset()

    self.batch_data_ = np.transpose(self.batch_data_, (1,3,4,0,2)).astype('float32') 
    self.batch_label_ = self.batch_label_.astype('int64')
    return self.batch_data_, self.batch_label_, n_examples
  
  '''
  def GetSingleExample(self, data_pb, idx, offset=0):
    ### length validation
    num_f = []
    for line in open(data_pb.num_frames_file):
      num_f.append(int(line.strip()))

    #if num_f[idx] < self.seq_length_:
    #    print 'Example is too short'
    #    exit()

    ### data_
    try:
      frames_before = np.cumsum(num_f[:idx],0)[-1]
    except IndexError:
      if idx==0:
        frames_before = 0
      else:
        frames_before = np.cumsum(num_f[:idx],0)[-1]
    start = frames_before + offset                 # inclusive
    end   = frames_before + num_f[idx] - 1         # inclusive
    length= num_f[idx] - offset
    skip = int(30.0/self.fps_)

    data_ = np.zeros((self.seq_length_, 1, self.frame_size_), dtype=np.float32)
    f = h5py.File(data_pb.data_file,'r')

    if length >= self.seq_length_*skip:
      data_[:,0,:] = np.concatenate(f[self.dataset_name_][start:start+self.seq_length_*skip:skip],axis=0)
    else:
      n = 1 + int((length-1)/skip)
      self.batch_data_[:n,0, :] = np.concatenate(f[self.dataset_name_][start:start+length:skip],axis=0)
      self.batch_data_[n:,0, :] = np.tile(self.batch_data_[n-1,0, :],(self.seq_length_-n,1))

    if data_pb.dataset=='ucf11':
      data_ = data_.reshape([data_.shape[0],data_.shape[1],49,1024],order='F').astype('float32')
    elif data_pb.dataset=='h2mAP':
      data_ = data_.reshape([data_.shape[0],data_.shape[1],49,1024],order='F').astype('float32')
    elif data_pb.dataset=='hmdb51gln':
      data_ = data_.reshape([data_.shape[0],data_.shape[1],49,1024],order='F').astype('float32')

    f.close()

    ### label_
    if data_pb.dataset!='h2mAP':
      labels = self.GetLabels(data_pb.labels_file)
      label  = labels[idx]
      label_ = np.zeros((self.seq_length_, 1), dtype=np.int64)
      label_[:,0] = np.tile(label,(1,self.seq_length_))
    else:
      labels = np.array(self.GetMAPLabels(data_pb.labels_file))
      label  = labels[idx,:]                              data.shape[1:4]       # (12,)
      label_ = np.zeros((self.seq_length_,1,12), dtype=np.int64) # (TS, 1, 12) # 12 classes in hollywood2
      label_[:,0,:] = np.tile(label,(self.seq_length_,1))
    assert len(num_f) == len(labels)

    ### fidx_
    fnames = []
    for line in open(data_pb.vid_name_file):
      fnames.append(line.strip())
    fidx_ = fnames[idx]

    return data_, label_, fidx_
  '''
  def GetBatchSize(self):
    return self.batch_size_

  def GetLabels(self, filename):
    labels = []
    if filename != '':
      for line in open(filename,'r'):
        labels.append(int(line.strip()))
    return labels
  '''
  def GetMAPLabels(self, filename):
    labels = []
    if filename != '':
      for line in open(filename,'r'):
        labels.append([int(x) for x in line.split(',')])
    return labels
  '''
  def GetDatasetSize(self):
    return self.dataset_size_

  def Reset(self):
    self.row_ = 0
    if self.randomize_:
      rng_state = np.random.get_state()
      np.random.shuffle(self.indices_)
      np.random.set_state(rng_state)
      np.random.shuffle(self.labels_)

class TrainProto(object):
  def __init__(self, bs, maxlen, stride, dataset, fps=30):
    self.seq_length = maxlen
    self.stride = stride
    self.randomize = True
    self.batch_size = bs
    self.dataset = dataset
    self.fps = fps
    if dataset=='ucf11':
      self.data_file       = os.path.join(videoDir, 'train_feats_ucf11.h5')
      #self.num_frames_file = os.path.join(videoDir, 'train_nums.txt')
      self.labels_file     = os.path.join(videoDir, 'train_labels_ucf11.txt')
      #self.vid_name_file   = os.path.join(videoDir, 'train_filename.txt')
      self.dataset_name    = 'feats'
    elif dataset=='h2mAP':
      self.data_file       = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='hmdb51gln':
      self.data_file       = '/ais/gobi3/u/shikhar/hmdb/dataset/train_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hmdb/dataset/train_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hmdb/dataset/train_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hmdb/dataset/train_filename.txt'
      self.dataset_name    = 'features'

class TestTrainProto(object):
  def __init__(self, bs, maxlen, stride, dataset, fps=30):
    self.seq_length = maxlen
    self.stride = stride
    self.randomize = False
    self.batch_size = bs
    self.dataset = dataset
    self.fps = fps
    if dataset=='ucf11':
      self.data_file       = os.path.join(videoDir, 'train_feats_ucf11.h5')
      #self.num_frames_file = os.path.join(videoDir, 'train_frame_nums.txt')
      self.labels_file     = os.path.join(videoDir, 'train_labels_ucf11.txt')
      #self.vid_name_file   = os.path.join(videoDir, 'train_filename.txt')
      self.dataset_name    = 'feats'
    elif dataset=='h2mAP':
      self.data_file       = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_framenum.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/train_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='hmdb51gln':
      self.data_file       = '/ais/gobi3/u/shikhar/hmdb/dataset/train_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hmdb/dataset/train_framenums.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hmdb/dataset/train_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hmdb/dataset/train_filename.txt'
      self.dataset_name    = 'features'

class TestValidProto(object):
  def __init__(self, bs, maxlen, stride, dataset, fps=30):
    self.seq_length = maxlen
    self.stride = stride
    self.randomize = False
    self.batch_size = bs
    self.dataset = dataset
    self.fps = fps
    if dataset=='ucf11':
      self.data_file       = os.path.join(videoDir, 'valid_feats_ucf11.h5')
      #self.num_frames_file = os.path.join(videoDir, 'valid_frame_nums.txt')
      self.labels_file     = os.path.join(videoDir, 'valid_labels_ucf11.txt')
      #self.vid_name_file   = os.path.join(videoDir, 'valid_filename.txt')
      self.dataset_name    = 'feats'
    elif dataset=='h2mAP':
      self.data_file       = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/valid_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/valid_framenums.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/valid_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/valid_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='hmdb51gln':
      self.data_file       = '/ais/gobi3/u/shikhar/hmdb/dataset/test_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hmdb/dataset/test_framenums.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hmdb/dataset/test_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hmdb/dataset/test_filename.txt'
      self.dataset_name    = 'features'

class TestTestProto(object):
  def __init__(self, bs, maxlen, stride, dataset, fps=30):
    self.seq_length = maxlen
    self.stride = stride
    self.randomize = False
    self.batch_size = bs
    self.dataset = dataset
    self.fps = fps
    if dataset=='ucf11':
      self.data_file       = os.path.join(videoDir, 'test_feats_ucf11.h5')
      #self.num_frames_file = os.path.join(videoDir, 'test_frame_nums.txt')
      self.labels_file     = os.path.join(videoDir, 'test_labels_ucf11.txt')
      #self.vid_name_file   = os.path.join(videoDir, 'test_filename.txt')
      self.dataset_name    = 'feats'
    elif dataset=='h2mAP':
      self.data_file       = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/test_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/test_framenums.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/test_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hollywood2/mAPdataset/test_filename.txt'
      self.dataset_name    = 'features'
    elif dataset=='hmdb51gln':
      self.data_file       = '/ais/gobi3/u/shikhar/hmdb/dataset/test_features.h5'
      self.num_frames_file = '/ais/gobi3/u/shikhar/hmdb/dataset/test_framenums.txt'
      self.labels_file     = '/ais/gobi3/u/shikhar/hmdb/dataset/test_labels.txt'
      self.vid_name_file   = '/ais/gobi3/u/shikhar/hmdb/dataset/test_filename.txt'
      self.dataset_name    = 'features'

def main():
  fps = 30
  data_pb = TrainProto(50,30,10,'ucf11',fps)
  dh = DataHandler(data_pb)
  start      = time.time()
  for i in xrange(dh.dataset_size_/dh.batch_size_):
    x,y,n_ex = dh.GetBatch(data_pb)
    print x.shape
    print y.shape
    print n_ex
    #exit()
  end        = time.time()
  print 'Duration', end-start
  x,y,n_ex = dh.GetBatch(data_pb)
  exit()

if __name__ == '__main__':
  main()

