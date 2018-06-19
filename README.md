## Action Recognition using Visual Attention

We propose a soft attention based model for the task of action recognition in videos. 
We use multi-layered Recurrent Neural Networks (RNNs) with Long-Short Term Memory 
(LSTM) units which are deep both spatially and temporally. Our model learns to focus 
selectively on parts of the video frames and classifies videos after taking a few 
glimpses. The model essentially learns which parts in the frames are relevant for the 
task at hand and attaches higher importance to them. We evaluate the model on UCF-11 
(YouTube Action), HMDB-51 and Hollywood2 datasets and analyze how the model focuses its 
attention depending on the scene and the action being performed.

## Dependencies

* Python 2.7
* [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [skimage](http://scikit-image.org/docs/dev/api/skimage.html)
* [Theano](http://www.deeplearning.net/software/theano/)
* [h5py](http://docs.h5py.org/en/latest/)


