# Data https://www.physionet.org/content/eegmmidb/1.0.0/
# Follow the instruction to download the data, 109 subjects, 64 channels, 160 Hz
# Download data to data/PhysioNet/eegmmidb/1.0.0

import os
import glob
import numpy as np
import mne
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

PATH = '../../rcnn/data/PhysioNet/eegmmidb/1.0.0/'
SUBS = glob.glob(PATH + 'S[0-9]*')
FNAMES = sorted(x[-4:] for x in SUBS)

# 1. Baseline, eyes open
# 2. Baseline, eyes closed
# 3. 7. 11.  Task 1 (open and close left or right fist)
# 5. 9. 13.  Task 3 (open and close both fists or both feet)
# 4. 8. 12.  Task 2 (imagine opening and closing left or right fist)
# 6. 10. 14. Task 4 (imagine opening and closing both fists or both feet)
#
# T0 corresponds to rest,
# T1 corresponds to onset of motion (real or imagined) of
#     the left fist  (in runs 3, 7, 11 for ME, and 4, 8, 12 for MI)
#     both fists     (in runs 5, 9, 13 for ME, and 6, 10, 14 for MI)
# T2 corresponds to onset of motion (real or imagined) of
#     the right fist (in runs 3, 7, 11 for ME, and 4, 8, 12 for MI)
#     both feet      (in runs 5, 9, 13 for ME, and 6, 10, 14 for MI)
FNAMES.remove('S089')


def get_data(subj_num=FNAMES, l_freq=13, h_freq=55, tmin=0.0, tmax=4.0):
    """ Extract EEG time series and class labels from edf files
    All T1, T2 events have duration 4.1 or 4.2 s, use 0~4s for epoching
    :param subj_num:
    :param l_freq: low frequency for bandpass filter
    :param h_freq: high frequency for bandpass filter
    :param tmin:
    :param tmax:

    :return: Xs, ys: list (Ni, nc, T), (Ni, )
    """

    run_type_0 = ['02']
    run_type_1 = ['04', '08', '12']
    run_type_2 = ['06', '10', '14']
    run_types = run_type_0 + run_type_1 + run_type_2

    # fixed parameters
    nc = 64       # number of channels
    sfreq = 160
    n_type0 = 22  # each subject has 21~24 fist or feet events, set to use the same number of data points for baseline
    event_id = {'T1': 2, 'T2': 3}  # T0:0 rest event, not interested

    T = round(tmax * sfreq)

    print('Sub, (N,   nc,  T)')
    Xs, ys = [], []

    # Iterate over subjects: S001, S002, S003 ...
    for i, subj in enumerate(subj_num):
        freq_flag = False

        # Get file names for motor imagery data
        fnames = glob.glob(os.path.join(PATH, subj, subj+'R*.edf'))
        fnames = sorted([name for name in fnames if name[-6:-4] in run_types])

        subject_trials, subject_labels = [], []
        # Iterate over subject's experiment runs
        for i, fname in enumerate(fnames):

            run = fname[-6:-4]

            # Load data into MNE raw object
            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)

            if raw.info['sfreq'] != 160:
                freq_flag = True
                print('{} is sampled at {}Hz so will be excluded.'.format(subj, raw.info['sfreq']))
                break

            # By default picks all 64 channels
            picks = mne.pick_types(raw.info, eeg=True)
            assert len(picks) == nc

            # Apply notch filter and bandpass filter
            raw.notch_filter([60.])
            raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, method='iir', verbose=False)

            if run == '02':
                # Baseline, eye closed
                # There is only the rest event, randomly slice n_type0=22 snippets of length T for this class
                data = raw.get_data() * 1e6
                X = np.zeros((n_type0, nc, T))
                y = np.zeros((n_type0, ), dtype=int)   # eye closed
                for i in range(n_type0):
                    offset = np.random.randint(0, data.shape[1] - T)
                    X[i] = data[:, offset: offset+T]

            else:
                # Epoching
                events, _ = mne.events_from_annotations(raw, verbose=False)
                epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None,
                                    # reject={'eeg': reject_threshold*1e-6},
                                    picks=picks, preload=True, verbose=False, on_missing='warning')

                # Data
                X = epochs.get_data() * 1e6     # converts to uV, the L503 in epochs.py makes T=641
                X = X[:, :, :-1]                # make T = sfeq * (tmax-tmin)

                # specific treatment of classes/labels
                y = epochs.events[:, -1]
                if run in run_type_1:
                    y[y == 2] = 1   # left fist
                    y[y == 3] = 2   # right fist
                else:  # run_type_2
                    y[y == 3] = 4   # both feet
                    y[y == 2] = 3   # both fists

            subject_trials.append(X)
            subject_labels.append(y)

        if freq_flag:
            continue
        subject_trials = np.concatenate(subject_trials, axis=0)  # (None, nc=64, T=640)
        subject_labels = np.concatenate(subject_labels)
        # print({i: sum(subject_labels == i) for i in range(5)})
        print(subj, subject_trials.shape)

        Xs.append(subject_trials)
        ys.append(subject_labels)

    return Xs, ys


def prepare_data(X, y, split_ratio=[0.64, 0.16, 0.2], chunking=True, set_seed=42):
    """ Prepare data for DNN, CNN, RNN models

    Partition data by subjects
    (Optional) Partition each trial of 4 sec to 10 consecutive chunks of 0.4 sec

    :param X:  list of (None, nc, T), each for a subject
    :param y:  list (None, )
    :param split_ratio: [train, val, test] split ratio
    :param chunking:  default False, if True, chunk T to 10 consecutive chunks of T/10

    :return: X_train, y_train, X_val, y_val, X_test, y_test
    """

    print('There are {} subjects, time series length {}, {} channels'.format(len(X), X[0].shape[1], X[0].shape[2]))

    X = np.concatenate(X, axis=0)

    # Eliminate the influence of reference electrode
    # here minus the average across all subjects and trials
    X = np.transpose(X, (0, 2, 1))
    N, T, nc = X.shape
    X = X.reshape(N*T, nc)
    X -= X.mean(axis=0)
    X = X.reshape(N, T, nc)
    X = np.transpose(X, (0, 2, 1))   # (N, nc, T)

    # Z-score for every trial for each time step
    scaler = StandardScaler()
    X = np.stack([scaler.fit_transform(x) for x in X])

    # Reshape to list of subjects
    cutoffs = np.cumsum([len(y_) for y_ in y])[:-1]
    X = np.array_split(X, cutoffs)  # [(N1, nc, T), (N2, nc, T), ...]

    # Train/val/test split by subjects
    np.random.seed(set_seed)
    shuffle_indices = np.random.permutation(len(X))
    test_size = int(len(X) * split_ratio[2])
    train_size = int(len(X) * split_ratio[0])

    X_train = [X[i] for i in shuffle_indices[:train_size]]
    y_train = [y[i] for i in shuffle_indices[:train_size]]
    X_val = [X[i] for i in shuffle_indices[train_size:-test_size]]
    y_val = [y[i] for i in shuffle_indices[train_size:-test_size]]
    X_test = [X[i] for i in shuffle_indices[-test_size:]]
    y_test = [y[i] for i in shuffle_indices[-test_size:]]

    def chunk(X, y):
        # Partition each trial of 4 sec to 10 consecutive chunks of 0.4 sec
        X_new, y_new = [], []
        for x, y_ in zip(X, y):
            x = np.array_split(x, 10, axis=-1)  # 10 x (N, nc, T/10)
            x = np.concatenate(x, axis=0)       # (10N, nc, T/10)
            X_new.append(x)

            y_ = y_.reshape(1, -1).repeat(10, axis=0).ravel()
            y_new.append(y_)

        return X_new, y_new

    if chunking:
        X_train, y_train = chunk(X_train, y_train)
        X_val, y_val = chunk(X_val, y_val)
        X_test, y_test = chunk(X_test, y_test)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train)
    X_train, y_train = shuffle(X_train, y_train)

    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val)
    X_val, y_val = shuffle(X_val, y_val)

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':

    X, y = get_data(subj_num=FNAMES, tmin=0.0, tmax=4.0)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        X, y, split_ratio=[0.64, 0.16, 0.2], chunking=False)
    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_val', X_val.shape)
    print('X_test', X_test.shape)

    data = {'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test}

    with open('../data/data.pkl', 'wb') as f:
        pickle.dump(data, f)
