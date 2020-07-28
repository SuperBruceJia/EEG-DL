#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

Please see this before you run this file!!!!!!

NOTICE:
    Be advised that this “Extract-Raw-Data-Into-Matlab-Files.py” Python File
    should only be executed under the Python 2 Environment.
    I highly recommend to execute the file under the Python 2.7 Environment
    because I have passed the test.
    However, if you are using Python 3 Environment to run this file,
    I'm afraid there will be an error and the generated labels will be wrong.

    If you have any question, please be sure to let me know.
    My email is shuyuej@ieee.org.
    Thanks a lot.

'''
import numpy as np
import io
import os
import pyedflib
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch, lfilter

MOVEMENT_START = 1 * 160  # MI starts 1s after trial begin
MOVEMENT_END   = 5 * 160  # MI lasts 4 seconds
NOISE_LEVEL    = 0.01

PHYSIONET_ELECTRODES = {
    1:  "FC5",  2: "FC3",  3: "FC1",  4: "FCz",  5: "FC2",  6: "FC4",
    7:  "FC6",  8: "C5",   9: "C3",  10: "C1",  11: "Cz",  12: "C2",
    13: "C4",  14: "C6",  15: "CP5", 16: "CP3", 17: "CP1", 18: "CPz",
    19: "CP2", 20: "CP4", 21: "CP6", 22: "Fp1", 23: "Fpz", 24: "Fp2",
    25: "AF7", 26: "AF3", 27: "AFz", 28: "AF4", 29: "AF8", 30: "F7",
    31: "F5",  32: "F3",  33: "F1",  34: "Fz",  35: "F2",  36: "F4",
    37: "F6",  38: "F8",  39: "FT7", 40: "FT8", 41: "T7",  42: "T8",
    43: "T9",  44: "T10", 45: "TP7", 46: "TP8", 47: "P7",  48: "P5",
    49: "P3",  50: "P1",  51: "Pz",  52: "P2",  53: "P4",  54: "P6",
    55: "P8",  56: "PO7", 57: "PO3", 58: "POz", 59: "PO4", 60: "PO8",
    61: "O1",  62: "Oz",  63: "O2",  64: "Iz"}


def load_edf_signals(path):
    try:
        sig = pyedflib.EdfReader(path)
        n = sig.signals_in_file
        signal_labels = sig.getSignalLabels()
        sigbuf = np.zeros((n, sig.getNSamples()[0]))

        for j in np.arange(n):
            sigbuf[j, :] = sig.readSignal(j)
        # (n,3) annotations: [t in s, duration, type T0/T1/T2]
        annotations = sig.read_annotation()
    except KeyboardInterrupt:
        # prevent memory leak and access problems of unclosed buffers
        sig._close()
        raise

    sig._close()
    del sig
    return sigbuf.transpose(), annotations


def get_physionet_electrode_positions():
    refpos = get_electrode_positions()
    return np.array([refpos[PHYSIONET_ELECTRODES[idx]] for idx in range(1, 65)])


def projection_2d(loc):
    """
    Azimuthal equidistant projection (AEP) of 3D carthesian coordinates.
    Preserves distance to origin while projecting to 2D carthesian space.
    loc: N x 3 array of 3D points
    returns: N x 2 array of projected 2D points
    """
    x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
    theta = np.arctan2(y, x)  # theta = azimuth
    rho = np.pi / 2 - np.arctan2(z, np.hypot(x, y))  # rho = pi/2 - elevation
    return np.stack((np.multiply(rho, np.cos(theta)), np.multiply(rho, np.sin(theta))), 1)


def get_electrode_positions():
    """
    Returns a dictionary (Name) -> (x,y,z) of electrode name in the extended
    10-20 system and its carthesian coordinates in unit sphere.
    """
    positions = dict()
    with io.open("electrode_positions.txt", "r") as pos_file:
        for line in pos_file:
            parts = line.split()
            positions[parts[0]] = tuple([float(part) for part in parts[1:]])
    return positions


def load_physionet_data(subject_id, num_classes=4, long_edge=False):
    """
    subject_id: ID (1-109) for the subject to be loaded from file
    num_classes: number of classes (2, 3 or 4) for L/R, L/R/0, L/R/0/F
    long_edge: if False include 1s before and after MI, if True include 3s
    returns (X, y, pos, fs)
        X: Trials with shape (N_subjects, N_trials, N_samples, N_channels)
        y: labels with shape (N_subjects, N_trials, N_classes)
        pos: 2D projected electrode positions
        fs: sample rate
    """

    SAMPLE_RATE  = 160
    EEG_CHANNELS = 64
    BASELINE_RUN = 1

    MI_RUNS = [4, 8, 12]  # l/r fist
    if num_classes >= 4:
        MI_RUNS += [6, 10, 14]  # feet (& fists)

    # total number of samples per long run
    RUN_LENGTH = 125 * SAMPLE_RATE

    # length of single trial in seconds
    TRIAL_LENGTH = 4 if not long_edge else 6
    NUM_TRIALS   = 21 * num_classes

    n_runs = len(MI_RUNS)
    X = np.zeros((n_runs, RUN_LENGTH, EEG_CHANNELS))
    events = []

    base_path = 'S%03dR%02d.edf'

    for i_run, current_run in enumerate(MI_RUNS):
        # load from file
        path = base_path % (subject_id, current_run)
        signals, annotations = load_edf_signals(path)
        X[i_run, :signals.shape[0], :] = signals

        # read annotations
        current_event = [i_run, 0, 0, 0]  # run, class (l/r), start, end

        for annotation in annotations:
            t = int(annotation[0] * SAMPLE_RATE * 1e-7)
            action = int(annotation[2][1])

            if action == 0 and current_event[1] != 0:
                # make 6 second runs by extending snippet
                length = TRIAL_LENGTH * SAMPLE_RATE
                pad = (length - (t - current_event[2])) / 2

                current_event[2] -= pad + (t - current_event[2]) % 2
                current_event[3] = t + pad

                if (current_run - 6) % 4 != 0 or current_event[1] == 2:
                    if (current_run - 6) % 4 == 0:
                        current_event[1] = 3
                    events.append(current_event)

            elif action > 0:
                current_event = [i_run, action, t, 0]
    
    # split runs into trials
    num_mi_trials = len(events)
    trials = np.zeros((NUM_TRIALS, TRIAL_LENGTH * SAMPLE_RATE, EEG_CHANNELS))
    labels = np.zeros((NUM_TRIALS, num_classes))

    for i, ev in enumerate(events):
        trials[i, :, :] = X[ev[0], ev[2]:ev[3]]
        labels[i, ev[1] - 1] = 1.

    if num_classes < 3:
        return (trials[:num_mi_trials, ...],
                labels[:num_mi_trials, ...],
                projection_2d(get_physionet_electrode_positions()),
                SAMPLE_RATE)
    else:
        # baseline run
        path = base_path % (subject_id, BASELINE_RUN)
        signals, annotations = load_edf_signals(path)
        SAMPLES = TRIAL_LENGTH * SAMPLE_RATE
        for i in range(num_mi_trials, NUM_TRIALS):
            offset = np.random.randint(0, signals.shape[0] - SAMPLES)
            trials[i, :, :] = signals[offset: offset+SAMPLES, :]
            labels[i, -1] = 1.
        return trials, labels, projection_2d(get_physionet_electrode_positions()), SAMPLE_RATE


def load_raw_data(electrodes, subject=None, num_classes=4, long_edge=False):
    # load from file
    trials = []
    labels = []

    if subject == None:
        subject_ids = range(11, 15)
    else:
        try:
            subject_ids = [int(subject)]
        except:
            subject_ids = subject

    for subject_id in subject_ids:
        try:
            t, l, loc, fs = load_physionet_data(subject_id, num_classes, long_edge=long_edge)
            if num_classes == 2 and t.shape[0] != 42:
                # drop subjects with less trials
                continue
            trials.append(t[:, :, electrodes])
            labels.append(l)
        except:
            pass
    return np.array(trials, dtype=np.float64).reshape((len(trials),) + trials[0].shape + (1,)), \
           np.array(labels, dtype=np.float64)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


print("Start to save the Files!")


# Sample rate and desired cutoff frequencies (in Hz).
fs      = 160.0
lowcut  = 5.0
highcut = 30.0

# Save Dataset of 4 clases
nclasses = 4

# Save 20 subjects' dataset
SAVE = '20-Subjects'
if not os.path.exists(SAVE):
    os.mkdir(SAVE)

subject = range(1, 21)

# Save 64 electrodes
for i in range(0, 64):
    electrodes = [i]
    X = 'X_' + str(i)
    Y = 'Y_' + str(i)
    X, Y = load_raw_data(electrodes=electrodes, subject=subject, num_classes=nclasses)
    X = np.squeeze(X)

    sio.savemat(SAVE + 'Dataset_%d.mat' % int(i+1), {'Dataset': X})
    sio.savemat(SAVE + 'Labels_%d.mat' % int(i+1), {'Labels': Y})

    print("Finished saving %d electrodes" % int(i+1))

# SAVE = '100-Subjects/'
#
# if not os.path.exists(SAVE):
#     os.mkdir(SAVE)
#
# subject = range(1, 102)
#
# for i in range(0, 64):
#     electrodes = [i]
#     X = 'X_' + str(i)
#     Y = 'Y_' + str(i)
#     X, Y = load_raw_data(electrodes=electrodes, subject=subject, num_classes=nclasses)
#     X = np.squeeze(X)
#
#     # # Notch Filter
#     # b, a = iirnotch(w0=60.0, Q=30.0, fs=fs)
#     # X = lfilter(b, a, X)
#     #
#     # # Butterworth Band-pass filter
#     # X = butter_bandpass_filter(X, lowcut, highcut, fs, order=4)
#
#     sio.savemat(SAVE + 'Dataset_%d.mat' % int(i+1), {'Dataset': X})
#     sio.savemat(SAVE + 'Labels_%d.mat' % int(i+1), {'Labels': Y})
#
#     print("Finished saving %d electrodes" % int(i+1))
