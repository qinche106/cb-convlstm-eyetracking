# cb-convlstm-eyetracking
This repo introduces you how to perform pupil detection using event stream from event-based cameras.
Here is our paper: 3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network, will be published at BioCAS 2023. The arxiv version portal: https://arxiv.org/pdf/2308.11771.pdf

Synthetic Event-based eye tracking (SEET) dataset link:
https://drive.google.com/drive/folders/16qH_wv_oVNysJARtHIUrIXbHjOygfq_i?usp=drive_link

Run steps:
1. Download the SEET dataset, and save it in a directory. i.e. /DATA/
2. cd eyetracking-convlstm
3. run process_event.py
4. run convlstm-et-pytorch-event.py

Results:
x, y coordinates of pupil center predictions after 28 epochs training
![我的图片描述](https://github.com/qinche106/cb-convlstm-eyetracking/blob/main/eyetracking-convlstm/plot/event_plot_28.png)

The accuracy reported in the BioCAS paper is not as high as the results demonstrated in this repository due to a discrepancy in the testing methodology. In the paper, a stride of 1 was maintained throughout the test set, which was not the optimal setting. It would be more appropriate to use a stride equivalent to the sequence length, especially when data augmentation should not be applied in the test set.

Original LPW dataset (not event-based dataset) Portal： 
https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/labelled-pupils-in-the-wild-lpw
