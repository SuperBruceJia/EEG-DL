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
<li><a href="#Research-Ideas">Research Ideas</a></li>
<li><a href="#Common-Issues">Common Issues</a></li>
<li><a href="#Structure-of-the-code">Structure of the code</a></li>
<li><a href="#Citation">Citation</a></li>
<li><a href="#Other-Useful-Resources">Other Useful Resources</a></li>
<li><a href="#Contribution">Contribution</a></li>
<li><a href="#Organizations">Organizations</a></li>
</ul>

## Documentation
**The supported models** include

| No.   | Model                                                  | Codes           |
| :----:| :----:                                                 | :----:          |
| 1     | Deep Neural Networks                                   | [DNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/DNN.py) |
| 2     | Convolutional Neural Networks [[Paper]](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta) [[Tutorial]](https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow)| [CNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/CNN.py) |
| 3     | Deep Residual Convolutional Neural Networks [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | [ResNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/ResCNN.py) |
| 4     | Thin Residual Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1902.10107) | [Thin ResNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/Thin_ResNet.py) |
| 5     | Densely Connected Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1608.06993) | [DenseNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/DenseCNN.py) |
| 6     | Fully Convolutional Neural Networks [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) | [FCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/Fully_Conv_CNN.py) |
| 7     | One Shot Learning with Siamese Networks (CNNs Backbone) <br> [[Paper]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) [[Tutorial]](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d) | [Siamese Networks](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/Siamese_Network.py) |
| 8     | Graph Convolutional Neural Networks <br> [[Paper]](https://ieeexplore.ieee.org/document/9889159) [[Presentation]](https://drive.google.com/file/d/1ecMbtZV2eH14sRAqWIIf1iRvDAC7DMDs/view?usp=sharing) [[Tutorial]](https://github.com/mdeff/cnn_graph) <br> [[GCN / GNN Summary for Chinese Readers]](https://github.com/wangyouze/GNN-algorithms) <br> [[GNN-related Algorithms Review for Chinese Readers]](https://github.com/LYuhang/GNN_Review) <br> [[Literature of Deep Learning for Graphs]](https://github.com/DeepGraphLearning/LiteratureDL4Graph) | [GCN / Graph CNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/GCN_Model.py) |
| 9     | Graph Convolutional Neural Networks <br> (Pure Python Implementation from [Reza Amini](https://github.com/magnumical)) | [GCN / Graph CNN](https://github.com/magnumical/GCN_for_EEG) |
| 10    | Deep Residual Graph Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/2007.13484) | [ResGCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/ResGCN_Model.py) | 
| 11    | Densely Connected Graph Convolutional Neural Networks  | [DenseGCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/DenseGCN_Model.py) |
| 12    | Bayesian Convolutional Neural Network <br> via Variational Inference <br> [[Paper]](https://arxiv.org/abs/1901.02731) [[Thesis]](https://github.com/kumar-shridhar/Master-Thesis-BayesianCNN/raw/master/thesis.pdf) <br> (PyTorch Implementation by [Kumar Shridhar](https://github.com/kumar-shridhar)) <br> [[Latest Codes]](https://github.com/kumar-shridhar/PyTorch-BayesianCNN) | [Bayesian CNNs](https://github.com/SuperBruceJia/EEG-BayesianCNN) |
| 13    | Recurrent Neural Networks [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [RNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/RNN.py) |
| 14    | Attention-based Recurrent Neural Networks [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [RNN with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/RNN_with_Attention.py) |
| 15    | Bidirectional Recurrent Neural Networks [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiRNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiRNN.py) |
| 16    | Attention-based Bidirectional Recurrent Neural Networks [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiRNN with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiRNN_with_Attention.py) |
| 17    | Long-short Term Memory [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [LSTM](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/LSTM.py) |
| 18    | Attention-based Long-short Term Memory [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [LSTM with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/LSTM_with_Attention.py) |
| 19    | Bidirectional Long-short Term Memory [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiLSTM](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiLSTM.py) |
| 20    | Attention-based Bidirectional Long-short Term Memory [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiLSTM with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiLSTM_with_Attention.py) |
| 21    | Gated Recurrent Unit [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [GRU](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/GRU.py) |
| 22    | Attention-based Gated Recurrent Unit [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [GRU with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/GRU_with_Attention.py) |
| 23    | Bidirectional Gated Recurrent Unit [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiGRU](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiGRU.py) |
| 24    | Attention-based Bidirectional Gated Recurrent Unit [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [BiGRU with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiGRU_with_Attention.py) |
| 25    | Attention-based BiLSTM + GCN [[Paper]](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full) | [Attention-based BiLSTM](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/BiLSTM_with_Attention.py) <br> [GCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Network/lib_for_GCN/GCN_Model.py) |
| 26    | Transformer [[Paper]](https://arxiv.org/abs/1706.03762) [[Paper]](https://arxiv.org/abs/2010.11929) | [Transformer](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/main-Transformer.py) |
| 26    | Transfer Learning with Transformer <br> (**This code is only for reference!**) <br> (**You can modify the codes to fit your applications.**) | Stage 1: [Pre-train](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/main-pretrain_model.py) <br> Stage 2: [Fine Tuning](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/main-finetuning_model.py) |

**One EEG Motor Imagery (MI) benchmark** is currently supported. Other benchmarks in the field of EEG or BCI can be found [here](https://github.com/meagmohit/EEG-Datasets).

| No.     | Dataset                                                                          | Tutorial |
| :----:  | :----:                                                                           | :----:   |
| 1       | [EEG Motor Movement/Imagery Dataset](https://archive.physionet.org/pn4/eegmmidb/) | [Tutorial](https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow)|

**The evaluation criteria** consists of

| Evaluation Metrics 					                                    | Tutorial |
| :----:                                                                    | :----:   |
| Confusion Matrix | [Tutorial](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) |
| Accuracy / Precision / Recall / F1 Score / Kappa Coefficient | [Tutorial](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) |
| Receiver Operating Characteristic (ROC) Curve / Area under the Curve (AUC)| - |
| Paired-wise t-test via R language | [Tutorial](https://www.analyticsvidhya.com/blog/2019/05/statistics-t-test-introduction-r-implementation/) |

*The evaluation metrics are mainly supported for **four-class classification**. If you wish to switch to two-class or three-class classification, please modify [this file](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/Evaluation_Metrics/Metrics.py) to adapt to your personal Dataset classes. Meanwhile, the details about the evaluation metrics can be found in [this paper](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta).*

## Usage Demo

1. ***(Under Any Python Environment)*** Download the [EEG Motor Movement/Imagery Dataset](https://archive.physionet.org/pn4/eegmmidb/) via [this script](https://github.com/SuperBruceJia/EEG-DL/blob/master/Download_Raw_EEG_Data/MIND_Get_EDF.py).

    ```text
    $ python MIND_Get_EDF.py
    ```

2. ***(Under Python 2.7 Environment)*** Read the .edf files (One of the raw EEG signals formats) and save them into Matlab .m files via [this script](https://github.com/SuperBruceJia/EEG-DL/blob/master/Download_Raw_EEG_Data/Extract-Raw-Data-Into-Matlab-Files.py). FYI, this script must be executed under the **Python 2 environment (Python 2.7 is recommended)** due to some Python 2 syntax. If using Python 3 environment to run the file, there might be no error, but the labels of EEG tasks would be totally messed up.

    ```text
    $ python Extract-Raw-Data-Into-Matlab-Files.py
    ```

3. Preprocessed the Dataset via the Matlab and save the data into the Excel files (training_set, training_label, test_set, and test_label) via [these scripts](https://github.com/SuperBruceJia/EEG-DL/tree/master/Preprocess_EEG_Data) with regards to different models. FYI, every lines of the Excel file is a sample, and the columns can be regarded as features, e.g., 4096 columns mean 64 channels X 64 time points. Later, the models will reshape 4096 columns into a Matrix with the shape 64 channels X 64 time points. You should can change the number of columns to fit your own needs, e.g., the real dimension of your own Dataset.

4. ***(Prerequsites)*** Train and test deep learning models **under the Python 3.6 Environment (Highly Recommended)** for EEG signals / tasks classification via [the EEG-DL library](https://github.com/SuperBruceJia/EEG-DL/tree/master/Models), which provides multiple SOTA DL models.

    ```text
    Python Version: Python 3.6 (Recommended)
    TensorFlow Version: TensorFlow 1.13.1
    ```

    Use the below command to install TensorFlow GPU Version 1.13.1:

    ```python
    $ pip install --upgrade --force-reinstall tensorflow-gpu==1.13.1 --user
    ```

5. Read evaluation criterias (through iterations) via the [Tensorboard](https://www.tensorflow.org/tensorboard). You can follow [this tutorial](https://www.guru99.com/tensorboard-tutorial.html). When you finished training the model, you will find the "events.out.tfevents.***" in the folder, e.g., "/Users/shuyuej/Desktop/trained_model/". You can use the following command in your terminal:

    ```python
    $ tensorboard --logdir="/Users/shuyuej/Desktop/trained_model/" --host=127.0.0.1
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

## Research Ideas
1. Dynamic Graph Convolutional Neural Networks [[Paper Survey]](https://github.com/SuperBruceJia/paper-reading/raw/master/Graph-Neural-Network/Dynamic-GCN-Survey.pptx) [[Paper Reading]](https://github.com/SuperBruceJia/paper-reading/tree/master/Graph-Neural-Network/Dynamic-GCN-Papers)

2. Neural Architecture Search / AutoML (Automatic Machine Learning) [[Tsinghua AutoGraph]](https://github.com/THUMNLab/AutoGL)

3. Reinforcement Learning Algorithms (e.g., Deep Q-Learning) [[Tsinghua Tianshou]](https://github.com/thu-ml/tianshou) [[Doc for Chinese Readers]](https://tianshou.readthedocs.io/zh/latest/docs/toc.html)

4. Bayesian Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1901.02731) [[Thesis]](https://github.com/kumar-shridhar/Master-Thesis-BayesianCNN/raw/master/thesis.pdf) [[Codes]](https://github.com/SuperBruceJia/EEG-BayesianCNN)

5. Transformer / Self-attention [[Paper Collections]](https://github.com/SuperBruceJia/paper-reading/tree/master/Machine-Learning/Transformer) [[Codes]](https://github.com/SuperBruceJia/EEG-DL/blob/master/Models/main-Transformer.py) [[Presentation]](https://github.com/SuperBruceJia/paper-reading/raw/master/paper-submiting/Towards%20Universal%20Models%20with%20NLP%20for%20Computer%20Vision%20Transformer%20and%20Attention%20Mechanism.pdf)

6. Self-supervised Learning (e.g., Contrastive Learning) [[Presentation]](https://github.com/SuperBruceJia/paper-reading/raw/master/paper-submiting/Self-Supervised%20Learning%20in%20Computer%20Vision-%20Past%2C%20Present%2C%20Trends.pdf)

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

5. **IndexError: list index out of range**
    
    To solve this issue, first of all, please double-check the Python Environment. ***Python 2.7 Environment is highly recommended.*** Besides, please install ***0.1.11*** version of ***pydeflib*** Python package. The install instruction is as follows:
    
    ```python
    $ pip install pyEDFlib==0.1.11
    ```

## Structure of the code

At the root of the project, you will see:

```text
├── Download_Raw_EEG_Data
│   ├── Extract-Raw-Data-Into-Matlab-Files.py
│   ├── MIND_Get_EDF.py
│   ├── README.md
│   └── electrode_positions.txt
├── Draw_Photos
│   ├── Draw_Accuracy_Photo.m
│   ├── Draw_Box_Photo.m
│   ├── Draw_Confusion_Matrix.py
│   ├── Draw_Loss_Photo.m
│   ├── Draw_ROC_and_AUC.py
│   └── figure_boxplot.m
├── LICENSE
├── Logo.png
├── MANIFEST.in
├── Models
│   ├── DatasetAPI
│   │   └── DataLoader.py
│   ├── Evaluation_Metrics
│   │   └── Metrics.py
│   ├── Initialize_Variables
│   │   └── Initialize.py
│   ├── Loss_Function
│   │   └── Loss.py
│   ├── Network
│   │   ├── BiGRU.py
│   │   ├── BiGRU_with_Attention.py
│   │   ├── BiLSTM.py
│   │   ├── BiLSTM_with_Attention.py
│   │   ├── BiRNN.py
│   │   ├── BiRNN_with_Attention.py
│   │   ├── CNN.py
│   │   ├── DNN.py
│   │   ├── DenseCNN.py
│   │   ├── Fully_Conv_CNN.py
│   │   ├── GRU.py
│   │   ├── GRU_with_Attention.py
│   │   ├── LSTM.py
│   │   ├── LSTM_with_Attention.py
│   │   ├── RNN.py
│   │   ├── RNN_with_Attention.py
│   │   ├── ResCNN.py
│   │   ├── Siamese_Network.py
│   │   ├── Thin_ResNet.py
│   │   └── lib_for_GCN
│   │       ├── DenseGCN_Model.py
│   │       ├── GCN_Model.py
│   │       ├── ResGCN_Model.py
│   │       ├── coarsening.py
│   │       └── graph.py
│   ├── __init__.py
│   ├── main-BiGRU-with-Attention.py
│   ├── main-BiGRU.py
│   ├── main-BiLSTM-with-Attention.py
│   ├── main-BiLSTM.py
│   ├── main-BiRNN-with-Attention.py
│   ├── main-BiRNN.py
│   ├── main-CNN.py
│   ├── main-DNN.py
│   ├── main-DenseCNN.py
│   ├── main-DenseGCN.py
│   ├── main-FullyConvCNN.py
│   ├── main-GCN.py
│   ├── main-GRU-with-Attention.py
│   ├── main-GRU.py
│   ├── main-LSTM-with-Attention.py
│   ├── main-LSTM.py
│   ├── main-RNN-with-Attention.py
│   ├── main-RNN.py
│   ├── main-ResCNN.py
│   ├── main-ResGCN.py
│   ├── main-Siamese-Network.py
│   └── main-Thin-ResNet.py
├── NEEPU.png
├── Preprocess_EEG_Data
│   ├── For-CNN-based-Models
│   │   └── make_dataset.m
│   ├── For-DNN-based-Models
│   │   └── make_dataset.m
│   ├── For-GCN-based-Models
│   │   └── make_dataset.m
│   ├── For-RNN-based-Models
│   │   └── make_dataset.m
│   └── For-Siamese-Network-One-Shot-Learning
│       └── make_dataset.m
├── README.md
├── Saved_Files
│   └── README.md
├── requirements.txt
└── setup.py
```

## Citation

If you find our library useful, please considering citing our paper in your publications.
We provide a BibTeX entry below.

```bibtex
@article{hou2022gcn,
	author={Hou, Yimin and Jia, Shuyue and Lun, Xiangmin and Hao, Ziqian and Shi, Yan and Li, Yang and Zeng, Rui and Lv, Jinglei},
	journal={IEEE Transactions on Neural Networks and Learning Systems}, 
	title={{GCNs-Net}: A Graph Convolutional Neural Network Approach for Decoding Time-Resolved EEG Motor Imagery Signals}, 
	volume={},
	number={},
	pages={1-12},
	year={Sept. 2022},
	doi={10.1109/TNNLS.2022.3202569}
}
  
@article{hou2020novel,
	title={A Novel Approach of Decoding EEG Four-class Motor Imagery Tasks via Scout ESI and CNN},
	author={Hou, Yimin and Zhou, Lu and Jia, Shuyue and Lun, Xiangmin},
	journal={Journal of Neural Engineering},
	volume={17},
	number={1},
	pages={016048},
	year={Feb. 2020},
	publisher={IOP Publishing},
	doi={10.1088/1741-2552/ab4af6}
	
}

@article{hou2022deep,
	author={Hou, Yimin and Jia, Shuyue and Lun, Xiangmin and Zhang, Shu and Chen, Tao and Wang, Fang and Lv, Jinglei},   
	title={Deep Feature Mining via the Attention-Based Bidirectional Long Short Term Memory Graph Convolutional Neural Network for Human Motor Imagery Recognition},
	journal={Frontiers in Bioengineering and Biotechnology},      
	volume={9},      
	year={Feb. 2022},      
	url={https://www.frontiersin.org/article/10.3389/fbioe.2021.706229},       
	doi={10.3389/fbioe.2021.706229},      
	ISSN={2296-4185}
}

@article{Jia2020AttentionGCN,
	title={Attention-based Graph ResNet for Motor Intent Detection from Raw EEG signals},
	author={Jia, Shuyue and Hou, Yimin and Lun, Xiangmin and Lv, Jinglei},
	journal={arXiv preprint arXiv:2007.13484},
	year={2022}
}
```

Our papers can be downloaded from:
1. [A Novel Approach of Decoding EEG Four-class Motor Imagery Tasks via Scout ESI and CNN](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta)<br>
*Codes and Tutorials for this work can be found [here](https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow).*<br>

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture1.jpg" alt="Project1">
</div>

--------------------------------------------------------------------------------

2. [GCNs-Net: A Graph Convolutional Neural Network Approach for Decoding Time-resolved EEG Motor Imagery Signals](https://ieeexplore.ieee.org/document/9889159)<br> 
*Presentation for this work can be found [here](https://drive.google.com/file/d/1ecMbtZV2eH14sRAqWIIf1iRvDAC7DMDs/view?usp=sharing).*<br>

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture2.png" alt="Project2">
</div>

--------------------------------------------------------------------------------

3. [Deep Feature Mining via Attention-based BiLSTM-GCN for Human Motor Imagery Recognition](https://www.frontiersin.org/articles/10.3389/fbioe.2021.706229/full)

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture3.jpg" alt="Project3.1">
</div>

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture4.jpg" alt="Project4.1">
</div>

--------------------------------------------------------------------------------

4. [Attention-based Graph ResNet for Motor Intent Detection from Raw EEG signals](https://arxiv.org/abs/2007.13484)

## Other Useful Resources

I think the following presentations could be helpful when you guys get engaged with Python & TensorFlow or build models.

1. Python Environment Setting-up Tutorial [download](https://github.com/SuperBruceJia/paper-reading/raw/master/other-presentations/Python-Environment-Set-up.pptx)

2. Usage of Cloud Server and Setting-up Tutorial [download](https://github.com/SuperBruceJia/paper-reading/raw/master/other-presentations/Usage%20of%20Server%20and%20Setting%20Up.pdf)

3. TensorFlow for Deep Learning Tutorial [download](https://github.com/SuperBruceJia/paper-reading/raw/master/other-presentations/TensorFlow-for-Deep-Learning.pdf)

## Contribution

We always welcome contributions to help make EEG-DL Library better. If you would like to contribute or have any question, please feel free to <a href="http://shuyuej.com/">contact me</a>, and my email is <a href="shuyuej@ieee.org">shuyuej@ieee.org</a>.

## Organizations

The library was created and open-sourced by Shuyue Jia, supervised by Prof. Yimin Hou, at the School of Automation Engineering, Northeast Electric Power University, Jilin, Jilin, China.<br>
<a href="http://www.neepu.edu.cn/"> <img width="500" height="150" src="https://github.com/SuperBruceJia/EEG-DL/raw/master/NEEPU.png"></a>
