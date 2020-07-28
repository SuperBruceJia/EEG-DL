For this file, we will download all the raw EEG MI data from the EEG Motor Movement/Imagery Dataset, you can visit [here](https://archive.physionet.org/pn4/eegmmidb/) to download them. Or, alternatively, you can use [MIND_Get_EDF.py](https://github.com/SuperBruceJia/EEG-DL/blob/master/Download_Raw_EEG_Data/EEG_Motor_Movement_Imagery_Dataset/MIND_Get_EDF.py) script to download all the files. The raw EEG signals will in .edf format. 

And then, we will use [Extract-Raw-Data-Into-Matlab-Files.py](https://github.com/SuperBruceJia/EEG-DL/blob/master/Download_Raw_EEG_Data/EEG_Motor_Movement_Imagery_Dataset/Extract-Raw-Data-Into-Matlab-Files.py) script to extract the raw data into .m Matlab files. Be advised that this “Extract-Raw-Data-Into-Matlab-Files.py” Python File should only be executed under the **Python 2 Environment**. I highly recommend to execute the file under the **Python 2.7 Environment** because I have passed the test. 

**However, if you are using Python 3 Environment to run this file, I'm afraid there will be a error and the generated labels will be wrong.**

```
Python Version: Python 2.7 (Highly Recommended)
```

Finally, we can preprocess the saved Matlab files using [these scripts](https://github.com/SuperBruceJia/EEG-DL/tree/master/Preprocess_EEG_Data).

If you have any question, please be sure to let me know. My email is shuyuej@ieee.org. Thanks a lot. 

