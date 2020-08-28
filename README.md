<p align="center">
  <a href="https://github.com/SuperBruceJia/EEG-DL"> <img width="500px" src="https://github.com/SuperBruceJia/EEG-DL/raw/master/Logo.png"></a> 
  <br />
  <br />
  <a href="https://gitter.im/EEG-DL/community"><img alt="Chat on Gitter" src="https://img.shields.io/gitter/room/nwjs/nw.js.svg" /></a>
  <a href="https://www.anaconda.com/"><img alt="Python Version" src="https://img.shields.io/badge/Python-3.x-green.svg" /></a>
  <a href="https://www.tensorflow.org/install"><img alt="TensorFlow Version" src="https://img.shields.io/badge/TensorFlow-1.13.1-red.svg" /></a>
  <a href="https://github.com/SuperBruceJia/EEG-DL/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>


<!-- <div align="center">
    <a href="https://github.com/SuperBruceJia/EEG-DL"> <img width="500px" src="https://github.com/SuperBruceJia/EEG-DL/raw/master/Logo.png"></a> 
</div> -->

--------------------------------------------------------------------------------

# Welcome to EEG Deep Learning Library

**EEG-DL** is a Deep Learning (DL) library written by [TensorFlow](https://www.tensorflow.org) for EEG Tasks (Signals) Classification. It provides the latest DL algorithms and keeps updated. 

<!-- [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/EEG-DL/community)
[![Python 3](https://img.shields.io/badge/Python-3.x-green.svg)](https://www.anaconda.com/)
[![TensorFlow 1.13.1](https://img.shields.io/badge/TensorFlow-1.13.1-red.svg)](https://www.tensorflow.org/install)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/SuperBruceJia/EEG-DL/blob/master/LICENSE) -->

## Table of Contents
<ul>
<li><a href="#Documentation">Documentation</a></li>
<li><a href="#Usage-Demo">Usage Demo</a></li>
<li><a href="#Notice">Notice</a></li>
<li><a href="#Common-Issues">Common Issues</a></li>
<li><a href="#Structure-of-the-code">Structure of the code</a></li>
<li><a href="#Citation">Citation</a></li>
<li><a href="#Contribution">Contribution</a></li>
<li><a href="#Organizations">Organizations</a></li>
</ul>

## Documentation
**The supported models** include

| No.   | Model                                                  | Codes           |
| :----:| :----:                                                 | :----:          |
| 1     | Deep Neural Networks                                   | [DNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/DNN.py) |
| 2     | Convolutional Neural Networks [[Paper]](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta) [[Tutorial]](https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow)| [CNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/CNN.py) |
| 3     | Deep Residual Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1512.03385) | [ResNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/ResCNN.py) |
| 4     | Thin Residual Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1902.10107) | [Thin ResNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/Thin_ResNet.py) |
| 5     | Densely Connected Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1608.06993) | [DenseNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/DenseCNN.py) |
| 6     | Fully Convolutional Neural Networks [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) | [FCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/Fully_Conv_CNN.py) |
| 7     | One Shot Learning with Siamese Networks (CNNs Backbone) <br> [[Paper]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) [[Tutorial]](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d) | [Siamese Networks](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/Siamese_Network.py) |
| 8     | Graph Convolutional Neural Networks <br> [[Paper]](https://arxiv.org/abs/2006.08924) [[Presentation]](https://drive.google.com/file/d/1ecMbtZV2eH14sRAqWIIf1iRvDAC7DMDs/view?usp=sharing) [[Tutorial]](https://github.com/mdeff/cnn_graph) | [GCN / Graph CNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/GCN_Model.py) |
| 9     | Graph Convolutional Neural Networks <br> (Pure Python Implementation from Reza Amini) | [GCN / Graph CNN](https://github.com/magnumical/GCN_for_EEG) |
| 10    | Deep Residual Graph Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/2007.13484) | [ResGCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/ResGCN_Model.py) | 
| 11    | Densely Connected Graph Convolutional Neural Networks  | [DenseGCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/DenseGCN_Model.py) |
| 12    | Recurrent Neural Networks [[Paper]](https://arxiv.org/abs/2005.00777) | [RNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/RNN.py) |
| 13    | Attention-based Recurrent Neural Networks [[Paper]](https://arxiv.org/abs/2005.00777) | [RNN with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/RNN_with_Attention.py) |
| 14    | Bidirectional Recurrent Neural Networks [[Paper]](https://arxiv.org/abs/2005.00777) | [BiRNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiRNN.py) |
| 15    | Attention-based Bidirectional Recurrent Neural Networks [[Paper]](https://arxiv.org/abs/2005.00777) | [BiRNN with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiRNN_with_Attention.py) |
| 16    | Long-short Term Memory [[Paper]](https://arxiv.org/abs/2005.00777) | [LSTM](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/LSTM.py) |
| 17    | Attention-based Long-short Term Memory [[Paper]](https://arxiv.org/abs/2005.00777) | [LSTM with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/LSTM_with_Attention.py) |
| 18    | Bidirectional Long-short Term Memory [[Paper]](https://arxiv.org/abs/2005.00777) | [BiLSTM](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiLSTM.py) |
| 19    | Attention-based Bidirectional Long-short Term Memory [[Paper]](https://arxiv.org/abs/2005.00777) | [BiLSTM with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiLSTM_with_Attention.py) |
| 20    | Gated Recurrent Unit [[Paper]](https://arxiv.org/abs/2005.00777) | [GRU](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/GRU.py) |
| 21    | Attention-based Gated Recurrent Unit [[Paper]](https://arxiv.org/abs/2005.00777) | [GRU with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/GRU_with_Attention.py) |
| 22    | Bidirectional Gated Recurrent Unit [[Paper]](https://arxiv.org/abs/2005.00777) | [BiGRU](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiGRU.py) |
| 23    | Attention-based Bidirectional Gated Recurrent Unit [[Paper]](https://arxiv.org/abs/2005.00777) | [BiGRU with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiGRU_with_Attention.py) |

**One EEG Motor Imagery (MI) benchmark** is currently supported. Other benchmarks in the field of EEG or BCI can be found [here](https://github.com/meagmohit/EEG-Datasets).

| No.     | Dataset                                                                          |
| :----:  | :----:                                                                           |
| 1       | [EEG Motor Movement/Imagery Dataset](https://archive.physionet.org/pn4/eegmmidb/) <br> [[Tutorial]](https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow)|

**The evaluation criteria** consists of

| Evaluation Metrics 					                                       |
| :----:                                                                    |
| Confusion Matrix [[Tutorial]](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) |
| Accuracy / Precision / Recall / F1 Score / Kappa Coefficient [[Tutorial]](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) |
| Receiver Operating Characteristic (ROC) Curve / Area under the Curve (AUC)|
| Paired-wise t-test (via R language [[Tutorial]](https://www.analyticsvidhya.com/blog/2019/05/statistics-t-test-introduction-r-implementation/)) |

*The evaluation metrics are mainly supported for **four-class classification**. If you wish to switch to two-class or three-class classification, please modify [this file](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Evaluation_Metrics/Metrics.py) to adapt to your personal Dataset classes. Meanwhile, the details about the evaluation metrics can be found in [this paper](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta).*

## Usage Demo

1. ***(Under Any Python Environment)*** Download the [EEG Motor Movement/Imagery Dataset](https://archive.physionet.org/pn4/eegmmidb/) via [this script](https://github.com/SuperBruceJia/EEG-DL/blob/master/Download_Raw_EEG_Data/MIND_Get_EDF.py).

    ```python
    $ python MIND_Get_EDF.py
    ```

2. ***(Under Python 2.7 Environment)*** Read the .edf files (One of the raw EEG signals formats) and save them into Matlab .m files via [this script](https://github.com/SuperBruceJia/EEG-DL/blob/master/Download_Raw_EEG_Data/Extract-Raw-Data-Into-Matlab-Files.py). FYI, this script must be executed under the **Python 2 environment (Python 2.7 is recommended)** due to some Python 2 syntax. If using Python 3 environment to run the file, there might be no error, but the labels of EEG tasks would be totally messed up.

    ```python
    $ python Extract-Raw-Data-Into-Matlab-Files.py
    ```

3. Preprocessed the Dataset via the Matlab and save the data into the Excel files (training_set, training_label, test_set, and test_label) via [these scripts](https://github.com/SuperBruceJia/EEG-DL/tree/master/Preprocess_EEG_Data) with regards to different models. FYI, every lines of the Excel file is a sample, and the columns can be regarded as features, e.g., 4096 columns mean 64 channels X 64 time points. Later, the models will reshape 4096 columns into a Matrix with the shape 64 channels X 64 time points. You should can change the number of columns to fit your own needs, e.g., the real dimension of your own Dataset.

4. ***(Prerequsites)*** Train and test deep learning models **under the Python 3.6 Environment (Highly Recommended)** for EEG signals / tasks classification via [the EEG-DL library](https://github.com/SuperBruceJia/EEG-DL/tree/master/Models), which provides multiple SOTA DL models.

    ```python
    Python Version: Python 3.6 (Recommended)
    TensorFlow Version: TensorFlow 1.13.1
    ```

    Use the below command to install TensorFlow GPU Version 1.13.1:

    ```python
    pip install --upgrade --force-reinstall tensorflow-gpu==1.13.1 --user
    ```

5. Read evaluation criterias (through iterations) via the [Tensorboard](https://www.tensorflow.org/tensorboard). You can follow [this tutorial](https://www.guru99.com/tensorboard-tutorial.html). When you finished training the model, you will find the "events.out.tfevents.***" in the folder, e.g., "/Users/shuyuej/Desktop/trained_model/". You can use the following command in your terminal:

    ```python
    tensorboard --logdir="/Users/shuyuej/Desktop/trained_model/" --host=127.0.0.1
    ```

    You can open the website in the [Google Chrome](https://www.google.com/chrome/) (Highly Recommended). 
    
    ```html
    http://127.0.0.1:6006/
    ```

    Then you can read and save the criterias into Excel .csv files.

6. Finally, draw beautiful paper photograph using Matlab or Python. Please follow [these scripts](https://github.com/SuperBruceJia/EEG-DL/tree/master/Draw_Photos).

## Notice
1. I have tested all the files (Python and Matlab) under the macOS. Be advised that for some Matlab files, several Matlab functions are different between Windows Operating System (OS) and macOS. For example, I used "readmatrix" function to read CSV files in the MacOS. However, I have to use “csvread” function in the Windows because there was no such "readmatrix" Matlab function in the Windows. If you have met similar problems, I recommend you to Google or Baidu them. You can definitely work them out.

2. For the GCNs-Net (GCN Model), for the graph Convolutional layer, the dimensionality of the graph will be unchanged, and for the max-pooling layer, the dimensionality of the graph will be reduced by 2. That means, if you have N X N graph Laplacian, after the max-pooling layer, the dimension will be N/2 X N/2. If you have a 15-channel EEG system, it cannot use max-pooling unless you selected 14 --> 7 or 12 --> 6 --> 3 or 10 --> 5 or 8 --> 4 --> 2 --> 1, etc. The details can be reviewed from [this paper](https://arxiv.org/abs/2006.08924).

3. The **Loss Function** can be changed or modified from [this file](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Loss_Function/Loss.py).

4. The **Dataset Loader** can be changed or modified from [this file](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/DatasetAPI/DataLoader.py).

## Common Issues
1. **ValueError: Cannot feed value of shape (1024, 1) for Tensor 'input/label:0', which has shape '(1024,)'**

    To solve this issue, you have to squeeze the shape of the labels from (1024, 1) to (1024,) using np.squeeze. Please edit the [DataLoader.py file](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/DatasetAPI/DataLoader.py).
    From original codes:
    ```python
    train_labels = pd.read_csv(DIR + 'training_label.csv', header=None)
    train_labels = np.array(train_labels).astype('float32')

    test_labels = pd.read_csv(DIR + 'test_label.csv', header=None)
    test_labels = np.array(test_labels).astype('float32')
    ```
    to
    ```python
    train_labels = pd.read_csv(DIR + 'training_label.csv', header=None)
    train_labels = np.array(train_labels).astype('float32')
    train_labels = np.squeeze(train_labels)

    test_labels = pd.read_csv(DIR + 'test_label.csv', header=None)
    test_labels = np.array(test_labels).astype('float32')
    test_labels = np.squeeze(test_labels)
    ```

2. **InvalidArgumentError: Nan in summary histogram for training/logits/bias/gradients**
    
    To solve this issue, you have to comment all the histogram summary. Please edit the [GCN_Model.py file](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/GCN_Model.py).

    ```python
    # Comment the above tf.summary.histogram from the GCN_Model.py File

    # # Histograms.
    # for grad, var in grads:
    #     if grad is None:
    #         print('warning: {} has no gradient'.format(var.op.name))
    #     else:
    #         tf.summary.histogram(var.op.name + '/gradients', grad)

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)
        return var
    ```

3. **TypeError: len() of unsized object**
    
    To solve this issue, you have to change the coarsen level to your own needs, and you can definitely change it to see the difference. Please edit the [main-GCN.py file](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/main-GCN.py). For example, if you want to implement the GCNs-Net to a 10-channel EEG system, you have to set "levels" equal to 1 or 0 because there is at most only one max-pooling (10 --> 5). And you can change argument "level" to 1 or 0 to see the difference.

    ```python
    # This is the coarsen levels, you can definitely change the level to observe the difference
    graphs, perm = coarsening.coarsen(Adjacency_Matrix, levels=5, self_connections=False)
    ```
    to
    ```python
    # This is the coarsen levels, you can definitely change the level to observe the difference
    graphs, perm = coarsening.coarsen(Adjacency_Matrix, levels=1, self_connections=False)
    ```

4. **tensorflow.python.framework.errors_impl.InvalidArgumentError: Received a label value of 7 which is outside the valid range of [0, 7).  Label values: 5 2 3 3 1 5 5 4 7 4 2 2 1 7 5 6 3 4 2 4**
    
    To solve this issue, for the GCNs-Net, when you make your dataset, you have to make your labels from 0 rather than 1. For example, if you have seven classes, your labels should be 0 (First class), 1 (Second class), 2 (Third class), 3 (Fourth class), 4 (Fifth class), 5 (Sixth class), 6 (Seventh class) instead of 1, 2, 3, 4, 5, 6, 7.

## Structure of the code

At the root of the project, you will see:

```text
├── Download_Raw_EEG_Data
|  └── Extract-Raw-Data-Into-Matlab-Files.py
|  └── MIND_Get_EDF.py
├── Preprocess_EEG_Data
|  └── For-CNN-based-Models
|  |  └── make_dataset.m
|  └── For-DNN-based-Models
|  |  └── make_dataset.m
|  └── For-GCN-based-Models
|  |  └── make_dataset.m
|  └── For-RNN-based-Models
|  |  └── make_dataset.m
|  └── For-Siamese-Network-One-Shot-Learning
|     └── make_dataset.m
├── Models
|  └── DatasetAPI
|  |  └── DataLoader.py
|  └── Evaluation_Metrics
|  |  └── Metrics.py
|  └── Initialize_Variables
|  |  └── Initialize.py
|  └── Loss_Function
|  |  └── Loss.py
|  └── Network
|  |  └── lib_for_GCN
|  |  |  └── coarsening.py
|  |  |  └── DenseGCN_Model.py
|  |  |  └── GCN_Model.py
|  |  |  └── graph.py
|  |  |  └── ResGCN_Model.py
|  |  └── DNN.py
|  |  └── CNN.py
|  |  └── ...
|  └── main-DNN.py
|  └── main-CNN.py
|  └── ...
├── Saved_Files
├── Draw_Photos
|  └── Draw_Accuracy_Photo.m
|  └── Draw_Box_Photo.m
|  └── Draw_Loss_Photo.m
|  └── figure_boxplot.m
|  └── Draw_Confusion_Matrix.py
|  └── Draw_ROC_and_AUC.py
```

## Citation

If you find our library useful, please considering citing our paper in your publications.
We provide a BibTeX entry below.

```bibtex
@article{hou2019novel,  
    year = 2020,  
    month = {feb},  
    publisher = {IOP Publishing},  
    volume = {17},  
    number = {1},  
    pages = {016048},  
    author = {Yimin Hou and Lu Zhou and Shuyue Jia and Xiangmin Lun},  
    title = {A novel approach of decoding {EEG} four-class motor imagery tasks via scout {ESI} and {CNN}},  
    journal = {Journal of Neural Engineering}  
}

@article{Lun2020GCNs,
  title={GCNs-Net: A Graph Convolutional Neural Network Approach for Decoding Time-resolved EEG Motor Imagery Signals},
  author={Lun, Xiangmin and Jia, Shuyue and Hou, Yimin and Shi, Yan and Li, Yang and Yang, Hanrui and Zhang, Shu and Lv, Jinglei},
  journal={arXiv preprint arXiv:2006.08924},
  year={2020}
}

@article{Hou2020DeepFM,
  title={Deep Feature Mining via Attention-based BiLSTM-GCN for Human Motor Imagery Recognition},
  author={Hou, Yimin and Jia, Shuyue and Zhang, Shu and Lun, Xiangmin and Shi, Yan and Li, Yang and Yang, Hanrui and Zeng, Rui and Lv, Jinglei},
  journal={arXiv preprint arXiv:2005.00777},
  year={2020}
}

@article{Jia2020AttentionGCN,
  title={Attention-based Graph ResNet for Motor Intent Detection from Raw EEG signals},
  author={Jia, Shuyue and Hou, Yimin and Shi, Yan and Li, Yang},
  journal={arXiv preprint arXiv:2007.13484},
  year={2020}
}
```

Our papers can be downloaded from:
1. [A Novel Approach of Decoding EEG Four-class Motor Imagery Tasks via Scout ESI and CNN](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta)<br>
*Codes and Tutorials for this work can be found [here](https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow).*<br>

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture2.png" alt="Project2">
</div>

--------------------------------------------------------------------------------

2. [GCNs-Net: A Graph Convolutional Neural Network Approach for Decoding Time-resolved EEG Motor Imagery Signals](https://arxiv.org/abs/2006.08924)<br> 
*Presentation for this work can be found [here](https://drive.google.com/file/d/1ecMbtZV2eH14sRAqWIIf1iRvDAC7DMDs/view?usp=sharing).*<br>

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture1.png" alt="Project2">
</div>

--------------------------------------------------------------------------------

3. [Deep Feature Mining via Attention-based BiLSTM-GCN for Human Motor Imagery Recognition](https://arxiv.org/abs/2005.00777)

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture4.png" alt="Project2">
</div>

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture5.png" alt="Project2">
</div>

--------------------------------------------------------------------------------

4. [Attention-based Graph ResNet for Motor Intent Detection from Raw EEG signals](https://arxiv.org/abs/2007.13484)

## Contribution

We always welcome contributions to help make EEG-DL Library better. If you would like to contribute or have any question, please feel free to <a href="http://shuyuej.com/">contact me</a>, and my email is <a href="shuyuej@ieee.org">shuyuej@ieee.org</a>.

## Organizations
The library was created and open-sourced by Shuyue Jia, supervised by Prof. Yimin Hou @ Human Sensor Laboratory, School of Automation Engineering, Northeast Electric Power University, Jilin, China.<br>
<a href="http://www.neepu.edu.cn/"> <img width="150" height="150" src="https://github.com/SuperBruceJia/EEG-DL/raw/master/NEEPU.png"></a>
