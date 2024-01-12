![](https://github.com/qinche106/cb-convlstm-eyetracking/blob/main/eyetracking-convlstm/plot/logo1.png)

3ET is an event-based eye-tracking dataset to facilitate the download!
💥💥💥 Loading 3ET dataset is now available with the Tonic event datasets loader [Jump to dataloader](#tonic-dataloader)
The Tonic data loader helps you to easily load raw events which can be further processed into your favorite event data representation including event frames.

If you find this repo helpful, please cite our paper. This is the archived version and the BioCAS proceeding version will follow.
```
@article{chen20233et,
  title={3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network},
  author={Chen, Qinyu and Wang, Zuowen and Liu, Shih-Chii and Gao, Chang},
  journal={arXiv preprint arXiv:2308.11771},
  year={2023}
}
```
# cb-convlstm-eyetracking
This repo introduces you to how to perform pupil detection using event stream from event-based cameras.
Here is our paper: 3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network, which was published at BioCAS 2023. The arxiv version: https://arxiv.org/pdf/2308.11771.pdf

Synthetic Event-based Eye Tracking (SEET) dataset link (event frame):
[Google Drive](https://drive.google.com/drive/folders/16qH_wv_oVNysJARtHIUrIXbHjOygfq_i?usp=sharing)

Synthetic Event-based Eye Tracking (SEET) dataset link (event raw data and .avi video files):
[Google Drive](https://drive.google.com/drive/folders/1HeOS5YBLruzHjwMKyBQfVTc_mJbsy_R1?usp=sharing)

Run steps:
1. Download the SEET dataset, and save it in a directory. i.e. /DATA/
2. cd eyetracking-convlstm
3. run process_event.py
   
   you can change the sequence length by setting the parameter *seq*  
4. run convlstm-et-pytorch-event.py

Results:
*x, y* coordinates of pupil center predictions after 28 epochs of training
![我的图片描述](https://github.com/qinche106/cb-convlstm-eyetracking/blob/main/eyetracking-convlstm/plot/event_plot_28.png)



Original LPW dataset (not event-based dataset)： 
https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/labelled-pupils-in-the-wild-lpw

<a id="tonic-dataloader"></a>
# Loading raw 3ET with the Tonic library 💥💥💥

We support using the Tonic library to automatically download the 3ET dataset and load the raw events and labels.

The Tonic library is a Python library for loading and processing event-based data. It is available at [GitHub Repo](https://github.com/neuromorphs/tonic/tree/develop)

To install the Tonic library branch which includes the 3ET dataset loader, run the following command:
```
pip install tonic --pre
```
A minimum working example of loading the 3ET dataset is shown below:
```
import tonic

trainset = tonic.datasets.ThreeET_Eyetracking(save_to="./data", split='Train')

events, targets = next(iter(trainset))
```

