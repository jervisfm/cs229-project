# cs229-project.

## TODO
- decide number of classes (try 50)
	- update script
- add visuals
	- <s>add tensorboard</s>
	- add models of binaries
- <s>constants file</s>
- make data splits (train, test, dev): DONE
- change eval metric (add speed)
- work on implementing SVM
	- try different kernels (use Skikit-learn)
	- figure out parameters

## Team Members
- Connie 
- Minh
- Jervis


## Introduction
Goal is to train a classifier that can recognize / classify handwritten digits. 

Dataset to use is the Google Quick Draw dataset.

## Downloading Dataset.

The Google quickdraw dataset is hosted on Google Cloud Storage.

To get a copy of it, you need to use gsutil to download.
* Install gsutil from here: https://cloud.google.com/storage/docs/gsutil_install#install
```
$ curl https://sdk.cloud.google.com | bash
$ exec -l $SHELL
$ gcloud init
```

* Download all the image drawings from the dataset.
```
$ ./download_fulldataset.sh
```

The fulldataset is big (~37GB). For initial testing, we would be using a small subset. 
```
$ ./download_minidataset.sh
```

## Randomly select categories from list.

We have a script that can randomly select categories to look at. To get 50 categories we run:
```
$ random_categories 50 
```

We also have a `download_projectdataset_50.sh` script that downloads the chosen 50 categories
for the project.

## Tensorboard Shortcuts
Run
```
tensorboard --logdir=logs/
```
Go to: http://localhost:6006

## Google cloud setup
We have a Google VM for our deep learning experiments. Our common user is cs229.

To access it run, the following command at the shell.
```
$ gcloud compute ssh --project cs229-2018 --zone "us-west1-b" cs229@cs229-vm-vm
```

You can set cs229-2018 as the default project for gcloud so you don't have to set it
each time by running
```
$ gcloud config set project cs229-2018
```

We now also have a GPU machine instance that we use for training. You can connect to it like so:

```
$ gcloud compute ssh --zone "us-west1-b" cs229@cs229-gpu-vm
```

For the regular CPU only version of the cloud VM, you can access it like so:

```
$ gcloud compute ssh --zone "us-west1-b" cs229@cs229-vm-vm
```

We also use GNU screen for session management. To check for list of available sessions
run 
```
$ screen -ls
```

We usually have a single `cs229` session that we all share. To attach to this session, just
run 
```
$ screen -x cs229
```

Some helpful screen commands:
* Open a new window in session - Ctrl + A, c
* Go to next window in session - Ctrl + A, n
* Go to previous window in session - Ctrl + A, p


## Initial Baseline
time: 56.0 seconds, epoch: 1, batch size: 500

```
Epoch 1/1
302140/302140 [==============================] - 6s 21us/step - loss: 0.8110 - acc: 0.5804
302141/302141 [==============================] - 4s 15us/step
Epoch 1/1
302141/302141 [==============================] - 6s 21us/step - loss: 1.0513 - acc: 0.5080
302140/302140 [==============================] - 5s 15us/step
Baseline: 60.44% (9.49%)
```

time: 103.0 seconds, epoch: 10, batch size: 500
```
Epoch 1/10
302140/302140 [==============================] - 8s 26us/step - loss: 0.8110 - acc: 0.5727
Epoch 2/10
302140/302140 [==============================] - 3s 9us/step - loss: 0.6897 - acc: 0.7010
Epoch 3/10
302140/302140 [==============================] - 3s 9us/step - loss: 0.6297 - acc: 0.7047
Epoch 4/10
302140/302140 [==============================] - 3s 10us/step - loss: 0.6001 - acc: 0.7049
Epoch 5/10
302140/302140 [==============================] - 3s 9us/step - loss: 0.5827 - acc: 0.7074
Epoch 6/10
302140/302140 [==============================] - 3s 9us/step - loss: 0.5740 - acc: 0.7066
Epoch 7/10
302140/302140 [==============================] - 3s 9us/step - loss: 0.5696 - acc: 0.7065
Epoch 8/10
302140/302140 [==============================] - 3s 9us/step - loss: 0.5689 - acc: 0.7062
Epoch 9/10
302140/302140 [==============================] - 3s 9us/step - loss: 0.5669 - acc: 0.7070
Epoch 10/10
302140/302140 [==============================] - 3s 9us/step - loss: 0.5651 - acc: 0.7084
302141/302141 [==============================] - 5s 16us/step
Epoch 1/10
302141/302141 [==============================] - 6s 20us/step - loss: 1.0515 - acc: 0.5082
Epoch 2/10
302141/302141 [==============================] - 3s 9us/step - loss: 1.0328 - acc: 0.5096
Epoch 3/10
302141/302141 [==============================] - 3s 9us/step - loss: 1.0327 - acc: 0.5096
Epoch 4/10
302141/302141 [==============================] - 3s 9us/step - loss: 1.0327 - acc: 0.5096
Epoch 5/10
302141/302141 [==============================] - 3s 9us/step - loss: 1.0327 - acc: 0.5096
Epoch 6/10
302141/302141 [==============================] - 3s 9us/step - loss: 1.0327 - acc: 0.5097
Epoch 7/10
302141/302141 [==============================] - 3s 9us/step - loss: 1.0327 - acc: 0.5097
Epoch 8/10
302141/302141 [==============================] - 3s 9us/step - loss: 1.0327 - acc: 0.5097
Epoch 9/10
302141/302141 [==============================] - 3s 8us/step - loss: 1.0327 - acc: 0.5097
Epoch 10/10
302141/302141 [==============================] - 2s 8us/step - loss: 1.0327 - acc: 0.5097
302140/302140 [==============================] - 5s 15us/step
Time elapsed:  103.03788709640503
Baseline: 61.02% (10.07%)
```

## SkLearn Baseline
Tested against 10 classes.
```
(finalProject) bash-3.2$ python baselinev2.py
/Users/jmuindi/miniconda3/envs/finalProject/lib/python3.6/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
All:  (1316034, 784)
train:  (1052824, 784)
Test:  (131607, 784)
Dev:  (131603, 784)
/Users/jmuindi/miniconda3/envs/finalProject/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:757: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
  "of iterations.", ConvergenceWarning)
Prediction:  ['squirrel' 'panda' 'squirrel' ... 'lighter' 'lighter' 'paint can']
Actual Label:  ['squirrel' 'squirrel' 'squirrel' ... 'lighter' 'lighter' 'lighter']
Accuracy:  0.6518316451752619
Confusion matrix:  [[ 6789    59   397    70   112   402   296  1542   794  1189]
 [  161 10019  1781   152    79    68   125    57   273   296]
 [  256  1641 12135   547   255   332   208   422  1253   398]
 [  129   134   244  9613   884   410   170   139   176   227]
 [  244   171   540  1013  8593   580   511   248   228   217]
 [  239    38   928   450   701  6556   434   560  1129   326]
 [ 1072   359   739   232   405  1143  6625   910  1232   751]
 [ 1590    16   511    22   215   389   407  7478   725   635]
 [  830   218   966   127   167   836   645   712 10863   324]
 [ 1628   180   997   194   210   462   570   782   384  7112]]
```

## SkLearn SVC
3 classes, 500 iters

Linear
```
Prediction: ['banana' 'eraser' 'eraser' ... 'banana' 'banana' 'belt']
Actual Label: ['banana' 'banana' 'banana' ... 'belt' 'belt' 'belt']
Acurracy: 0.407952380952
Training time(secs): 204.221334934
('Labels: ', ['banana', 'eraser', 'belt'])
('Confusion matrix: ', array([[3437, 3432,  131],
       [2728, 2942, 1330],
       [3360, 1452, 2188]]))
Confusion matrix, without normalization
[[3437 3432  131]
 [2728 2942 1330]
 [3360 1452 2188]]
Normalized confusion matrix
[[0.49 0.49 0.02]
 [0.39 0.42 0.19]
 [0.48 0.21 0.31]]
```
```
Acurracy: 0.222157142857
Training time(secs): 1831.04826593
('Labels: ', ['laptop', 'paint can', 'wristwatch', 'rain', 'panda', 'The Mona Lisa', 'nose', 'pond', 'hockey stick', 'banana'])
('Confusion matrix: ', array([[2523,  127, 1612,  510,  216,  267,  580,  420,   76,  669],
       [ 295,  694, 1761,  404,  377,  878, 1739,  283,  234,  335],
       [ 334,  277,  722,  205,  268, 2804,  781,  366,  344,  899],
       [ 279,  217,  546,  639,  117, 1017,  731,  284,  274, 2896],
       [ 365,  574, 1609,  643, 1608,  734,  876,  447,   40,  104],
       [ 419,  577, 2115,  735,  501, 1849,  462,  244,   63,   35],
       [ 268,   38,  179,   53,   20,  227,  901,  155,  909, 4250],
       [ 502,  278,  290,  371,  394, 1171,  851, 1791,  256, 1096],
       [ 226,   23,  112,   30,   16,  217,  955,   73, 1429, 3919],
       [ 923,   50,  380,  146,   50,  317,  886,  216,  637, 3395]]))
Confusion matrix, without normalization
[[2523  127 1612  510  216  267  580  420   76  669]
 [ 295  694 1761  404  377  878 1739  283  234  335]
 [ 334  277  722  205  268 2804  781  366  344  899]
 [ 279  217  546  639  117 1017  731  284  274 2896]
 [ 365  574 1609  643 1608  734  876  447   40  104]
 [ 419  577 2115  735  501 1849  462  244   63   35]
 [ 268   38  179   53   20  227  901  155  909 4250]
 [ 502  278  290  371  394 1171  851 1791  256 1096]
 [ 226   23  112   30   16  217  955   73 1429 3919]
 [ 923   50  380  146   50  317  886  216  637 3395]]
Normalized confusion matrix
[[0.36 0.02 0.23 0.07 0.03 0.04 0.08 0.06 0.01 0.1 ]
 [0.04 0.1  0.25 0.06 0.05 0.13 0.25 0.04 0.03 0.05]
 [0.05 0.04 0.1  0.03 0.04 0.4  0.11 0.05 0.05 0.13]
 [0.04 0.03 0.08 0.09 0.02 0.15 0.1  0.04 0.04 0.41]
 [0.05 0.08 0.23 0.09 0.23 0.1  0.13 0.06 0.01 0.01]
 [0.06 0.08 0.3  0.1  0.07 0.26 0.07 0.03 0.01 0.01]
 [0.04 0.01 0.03 0.01 0.   0.03 0.13 0.02 0.13 0.61]
 [0.07 0.04 0.04 0.05 0.06 0.17 0.12 0.26 0.04 0.16]
 [0.03 0.   0.02 0.   0.   0.03 0.14 0.01 0.2  0.56]
 [0.13 0.01 0.05 0.02 0.01 0.05 0.13 0.03 0.09 0.48]]
```

Polynomial
```
Prediction: ['banana' 'banana' 'banana' ... 'belt' 'belt' 'eraser']
Actual Label: ['banana' 'banana' 'banana' ... 'belt' 'belt' 'belt']
Acurracy: 0.516952380952
Training time(secs): 447.517017126
('Labels: ', ['banana', 'eraser', 'belt'])
('Confusion matrix: ', array([[5492,  260, 1248],
       [3976, 1482, 1542],
       [2301,  817, 3882]]))
Confusion matrix, without normalization
[[5492  260 1248]
 [3976 1482 1542]
 [2301  817 3882]]
Normalized confusion matrix
[[0.78 0.04 0.18]
 [0.57 0.21 0.22]
 [0.33 0.12 0.55]]
```
```
Prediction: ['banana' 'hockey stick' 'banana' ... 'pond' 'The Mona Lisa' 'pond']
Actual Label: ['banana' 'banana' 'banana' ... 'wristwatch' 'wristwatch' 'wristwatch']
Acurracy: 0.508942857143
Training time(secs): 6672.996176
('Labels: ', ['laptop', 'paint can', 'wristwatch', 'rain', 'panda', 'The Mona Lisa', 'nose', 'pond', 'hockey stick', 'banana'])
('Confusion matrix: ', array([[3912,  123,    6,   15,   98, 1274,   29,   88,  105, 1350],
       [ 227, 2974,   24,    4,  203, 1820,  180,   69,   60, 1439],
       [ 132,  218, 2527,   96,  498,  545,  269,  781,  432, 1502],
       [ 214,  236,   53, 1495,  399, 1116,  634,  639,  864, 1350],
       [ 325,  277,   17,   42, 2478,  992,  127,   69,   52, 2621],
       [ 302,  287,    8,   24,  250, 5493,   32,   18,   28,  558],
       [ 132,   66,   15,   66,   53,  274, 3589,  154, 1601, 1050],
       [ 465,  181,  133,   74,  660,  263,  161, 4148,  101,  814],
       [  61,   37,    4,    9,   17,  125,  296,   37, 3856, 2558],
       [ 131,   45,   16,   33,   54,  164,  166,  135, 1102, 5154]]))
Confusion matrix, without normalization
[[3912  123    6   15   98 1274   29   88  105 1350]
 [ 227 2974   24    4  203 1820  180   69   60 1439]
 [ 132  218 2527   96  498  545  269  781  432 1502]
 [ 214  236   53 1495  399 1116  634  639  864 1350]
 [ 325  277   17   42 2478  992  127   69   52 2621]
 [ 302  287    8   24  250 5493   32   18   28  558]
 [ 132   66   15   66   53  274 3589  154 1601 1050]
 [ 465  181  133   74  660  263  161 4148  101  814]
 [  61   37    4    9   17  125  296   37 3856 2558]
 [ 131   45   16   33   54  164  166  135 1102 5154]]
Normalized confusion matrix
[[5.59e-01 1.76e-02 8.57e-04 2.14e-03 1.40e-02 1.82e-01 4.14e-03 1.26e-02
  1.50e-02 1.93e-01]
 [3.24e-02 4.25e-01 3.43e-03 5.71e-04 2.90e-02 2.60e-01 2.57e-02 9.86e-03
  8.57e-03 2.06e-01]
 [1.89e-02 3.11e-02 3.61e-01 1.37e-02 7.11e-02 7.79e-02 3.84e-02 1.12e-01
  6.17e-02 2.15e-01]
 [3.06e-02 3.37e-02 7.57e-03 2.14e-01 5.70e-02 1.59e-01 9.06e-02 9.13e-02
  1.23e-01 1.93e-01]
 [4.64e-02 3.96e-02 2.43e-03 6.00e-03 3.54e-01 1.42e-01 1.81e-02 9.86e-03
  7.43e-03 3.74e-01]
 [4.31e-02 4.10e-02 1.14e-03 3.43e-03 3.57e-02 7.85e-01 4.57e-03 2.57e-03
  4.00e-03 7.97e-02]
 [1.89e-02 9.43e-03 2.14e-03 9.43e-03 7.57e-03 3.91e-02 5.13e-01 2.20e-02
  2.29e-01 1.50e-01]
 [6.64e-02 2.59e-02 1.90e-02 1.06e-02 9.43e-02 3.76e-02 2.30e-02 5.93e-01
  1.44e-02 1.16e-01]
 [8.71e-03 5.29e-03 5.71e-04 1.29e-03 2.43e-03 1.79e-02 4.23e-02 5.29e-03
  5.51e-01 3.65e-01]
 [1.87e-02 6.43e-03 2.29e-03 4.71e-03 7.71e-03 2.34e-02 2.37e-02 1.93e-02
  1.57e-01 7.36e-01]]
```

RBF
```
Prediction: ['banana' 'banana' 'banana' ... 'belt' 'eraser' 'belt']
Actual Label: ['banana' 'banana' 'banana' ... 'belt' 'belt' 'belt']
Acurracy: 0.62680952381
Training time(secs): 316.684726
('Labels: ', ['banana', 'eraser', 'belt'])
('Confusion matrix: ', array([[6657,    1,  342],
       [3664,   89, 3247],
       [ 541,   42, 6417]]))
Confusion matrix, without normalization
[[6657    1  342]
 [3664   89 3247]
 [ 541   42 6417]]
Normalized confusion matrix
[[9.51e-01 1.43e-04 4.89e-02]
 [5.23e-01 1.27e-02 4.64e-01]
 [7.73e-02 6.00e-03 9.17e-01]]
```
```
Prediction: ['banana' 'hockey stick' 'banana' ... 'hockey stick' 'hockey stick'
 'wristwatch']
Actual Label: ['banana' 'banana' 'banana' ... 'wristwatch' 'wristwatch' 'wristwatch']
Acurracy: 0.610128571429
Training time(secs): 2841.77415299
('Labels: ', ['laptop', 'paint can', 'wristwatch', 'rain', 'panda', 'The Mona Lisa', 'nose', 'pond', 'hockey stick', 'banana'])
('Confusion matrix: ', array([[5727,  208,   14,  104,  100,  230,   54,   42,  178,  343],
       [ 234, 5500,   36,  222,  168,  243,  210,   88,  172,  127],
       [ 162,  338, 1828, 1596,  358,  119,  227,  599, 1160,  613],
       [ 162,   96,   19, 5908,   81,   67,   63,  168,  161,  275],
       [ 296,  642,   70,  350, 4491,  275,  297,  142,  151,  286],
       [ 462,  819,   20,  142,  190, 5108,   51,   43,   79,   86],
       [  83,   41,   22, 2647,   36,   45,  713,   59, 2600,  754],
       [1073,  128,  137,  354,  469,   89, 1114, 1985,  103, 1548],
       [  38,   10,   16,  173,   18,   22,   34,   28, 5773,  888],
       [  41,   12,   33,  191,   37,   25,   50,   48,  887, 5676]]))
Confusion matrix, without normalization
[[5727  208   14  104  100  230   54   42  178  343]
 [ 234 5500   36  222  168  243  210   88  172  127]
 [ 162  338 1828 1596  358  119  227  599 1160  613]
 [ 162   96   19 5908   81   67   63  168  161  275]
 [ 296  642   70  350 4491  275  297  142  151  286]
 [ 462  819   20  142  190 5108   51   43   79   86]
 [  83   41   22 2647   36   45  713   59 2600  754]
 [1073  128  137  354  469   89 1114 1985  103 1548]
 [  38   10   16  173   18   22   34   28 5773  888]
 [  41   12   33  191   37   25   50   48  887 5676]]
Normalized confusion matrix
[[0.82 0.03 0.   0.01 0.01 0.03 0.01 0.01 0.03 0.05]
 [0.03 0.79 0.01 0.03 0.02 0.03 0.03 0.01 0.02 0.02]
 [0.02 0.05 0.26 0.23 0.05 0.02 0.03 0.09 0.17 0.09]
 [0.02 0.01 0.   0.84 0.01 0.01 0.01 0.02 0.02 0.04]
 [0.04 0.09 0.01 0.05 0.64 0.04 0.04 0.02 0.02 0.04]
 [0.07 0.12 0.   0.02 0.03 0.73 0.01 0.01 0.01 0.01]
 [0.01 0.01 0.   0.38 0.01 0.01 0.1  0.01 0.37 0.11]
 [0.15 0.02 0.02 0.05 0.07 0.01 0.16 0.28 0.01 0.22]
 [0.01 0.   0.   0.02 0.   0.   0.   0.   0.82 0.13]
 [0.01 0.   0.   0.03 0.01 0.   0.01 0.01 0.13 0.81]]
```

Sigmoid
```
Prediction: ['eraser' 'eraser' 'eraser' ... 'eraser' 'eraser' 'eraser']
Actual Label: ['banana' 'banana' 'banana' ... 'belt' 'belt' 'belt']
Acurracy: 0.330761904762
Training time(secs): 478.207917929
('Labels: ', ['banana', 'eraser', 'belt'])
('Confusion matrix: ', array([[   0, 6589,  411],
       [   2, 6916,   82],
       [   2, 6968,   30]]))
Confusion matrix, without normalization
[[   0 6589  411]
 [   2 6916   82]
 [   2 6968   30]]
Normalized confusion matrix
[[0.00e+00 9.41e-01 5.87e-02]
 [2.86e-04 9.88e-01 1.17e-02]
 [2.86e-04 9.95e-01 4.29e-03]]
```
```
Prediction: ['wristwatch' 'wristwatch' 'rain' ... 'wristwatch' 'wristwatch'
 'wristwatch']
Actual Label: ['banana' 'banana' 'banana' ... 'wristwatch' 'wristwatch' 'wristwatch']
Acurracy: 0.117157142857
Training time(secs): 6971.01470613
('Labels: ', ['laptop', 'paint can', 'wristwatch', 'rain', 'panda', 'The Mona Lisa', 'nose', 'pond', 'hockey stick', 'banana'])
('Confusion matrix: ', array([[   2,    5, 6792,   75,   26,   21,    2,   69,    6,    2],
       [   0,    0, 6242,  724,    4,   17,    0,   13,    0,    0],
       [   1,    0, 6606,  381,    1,    8,    0,    2,    1,    0],
       [   0,    0, 5155, 1544,    1,  155,  122,    6,   17,    0],
       [   3,    0, 6887,   56,   16,   18,    0,   20,    0,    0],
       [   3,    5, 6810,   42,   35,   24,    0,   81,    0,    0],
       [   0,    0, 2556, 4414,    0,   23,    3,    3,    1,    0],
       [   0,    1, 6920,   53,    2,   17,    0,    6,    1,    0],
       [   1,    0, 3215, 3754,    0,   17,   12,    1,    0,    0],
       [   0,    0, 5414, 1565,    0,   12,    9,    0,    0,    0]]))
Confusion matrix, without normalization
[[   2    5 6792   75   26   21    2   69    6    2]
 [   0    0 6242  724    4   17    0   13    0    0]
 [   1    0 6606  381    1    8    0    2    1    0]
 [   0    0 5155 1544    1  155  122    6   17    0]
 [   3    0 6887   56   16   18    0   20    0    0]
 [   3    5 6810   42   35   24    0   81    0    0]
 [   0    0 2556 4414    0   23    3    3    1    0]
 [   0    1 6920   53    2   17    0    6    1    0]
 [   1    0 3215 3754    0   17   12    1    0    0]
 [   0    0 5414 1565    0   12    9    0    0    0]]
Normalized confusion matrix
[[2.86e-04 7.14e-04 9.70e-01 1.07e-02 3.71e-03 3.00e-03 2.86e-04 9.86e-03
  8.57e-04 2.86e-04]
 [0.00e+00 0.00e+00 8.92e-01 1.03e-01 5.71e-04 2.43e-03 0.00e+00 1.86e-03
  0.00e+00 0.00e+00]
 [1.43e-04 0.00e+00 9.44e-01 5.44e-02 1.43e-04 1.14e-03 0.00e+00 2.86e-04
  1.43e-04 0.00e+00]
 [0.00e+00 0.00e+00 7.36e-01 2.21e-01 1.43e-04 2.21e-02 1.74e-02 8.57e-04
  2.43e-03 0.00e+00]
 [4.29e-04 0.00e+00 9.84e-01 8.00e-03 2.29e-03 2.57e-03 0.00e+00 2.86e-03
  0.00e+00 0.00e+00]
 [4.29e-04 7.14e-04 9.73e-01 6.00e-03 5.00e-03 3.43e-03 0.00e+00 1.16e-02
  0.00e+00 0.00e+00]
 [0.00e+00 0.00e+00 3.65e-01 6.31e-01 0.00e+00 3.29e-03 4.29e-04 4.29e-04
  1.43e-04 0.00e+00]
 [0.00e+00 1.43e-04 9.89e-01 7.57e-03 2.86e-04 2.43e-03 0.00e+00 8.57e-04
  1.43e-04 0.00e+00]
 [1.43e-04 0.00e+00 4.59e-01 5.36e-01 0.00e+00 2.43e-03 1.71e-03 1.43e-04
  0.00e+00 0.00e+00]
 [0.00e+00 0.00e+00 7.73e-01 2.24e-01 0.00e+00 1.71e-03 1.29e-03 0.00e+00
  0.00e+00 0.00e+00]]
```

## CNN initial test run

### CNN 50 classes trial.

```
Epoch 95/100
2800000/2800000 [==============================] - 273s 97us/step - loss: 0.6510 - acc: 0.8326
Epoch 96/100
2800000/2800000 [==============================] - 278s 99us/step - loss: 0.6505 - acc: 0.8329
Epoch 97/100
2800000/2800000 [==============================] - 280s 100us/step - loss: 0.6504 - acc: 0.8327
Epoch 98/100
2800000/2800000 [==============================] - 277s 99us/step - loss: 0.6505 - acc: 0.8327
Epoch 99/100
2800000/2800000 [==============================] - 273s 97us/step - loss: 0.6507 - acc: 0.8327
Epoch 100/100
2800000/2800000 [==============================] - 275s 98us/step - loss: 0.6505 - acc: 0.8328
('Cnn model metrics', ['loss', 'acc'])
acc: 83.09%
Saved model to disk
Done building estimator
('Time elapsed: ', 27873.78034901619)
('Dummy y pred dev class', 31)
('Confusion matrix', array([[6375,   15,    6, ...,    5,    8,   20],
       [  12, 3760,   85, ...,  204,    2,   41],
       [   9,   60, 6432, ...,   15,    5,   10],
       ...,
       [   1,  149,    9, ..., 5643,    5,   49],
       [   1,   12,    3, ...,    1, 6435,    3],
       [  17,   56,   15, ...,   64,    7, 5282]]))
```

### 300 class test run
The simple model run did not complete fully and was terminated due to taking too long to run.

#### Modelv1
This was started at around 2018-12-05 22:13:47.545576. Been running for 12+ hours
Epoch 1/100
2018-12-05 22:13:46.985953:

```
Epoch 82/100
4800000/4800000 [==============================] - 569s 118us/step - loss: 1.6680 - acc: 0.6064
Epoch 83/100
4800000/4800000 [==============================] - 568s 118us/step - loss: 1.6675 - acc: 0.6064
Epoch 84/100
4800000/4800000 [==============================] - 570s 119us/step - loss: 1.6673 - acc: 0.6065
Epoch 85/100
4800000/4800000 [==============================] - 572s 119us/step - loss: 1.6664 - acc: 0.6067
Epoch 86/100
4800000/4800000 [==============================] - 571s 119us/step - loss: 1.6668 - acc: 0.6065
Epoch 87/100
4800000/4800000 [==============================] - 568s 118us/step - loss: 1.6670 - acc: 0.6066
Epoch 88/100
4800000/4800000 [==============================] - 566s 118us/step - loss: 1.6671 - acc: 0.6064
Epoch 89/100
4800000/4800000 [==============================] - 566s 118us/step - loss: 1.6664 - acc: 0.6066
Epoch 90/100
4800000/4800000 [==============================] - 565s 118us/step - loss: 1.6664 - acc: 0.6066
Epoch 91/100
4800000/4800000 [==============================] - 566s 118us/step - loss: 1.6660 - acc: 0.6068
Epoch 92/100
4800000/4800000 [==============================] - 569s 119us/step - loss: 1.6660 - acc: 0.6069
Epoch 93/100
4800000/4800000 [==============================] - 572s 119us/step - loss: 1.6656 - acc: 0.6067
Epoch 94/100
4800000/4800000 [==============================] - 570s 119us/step - loss: 1.6663 - acc: 0.6068
Epoch 95/100
4800000/4800000 [==============================] - 569s 119us/step - loss: 1.6656 - acc: 0.6067
Epoch 96/100
4800000/4800000 [==============================] - 3392s 707us/step - loss: 1.6656 - acc: 0.6069
Epoch 97/100
4370000/4800000 [==========================>...] - ETA: 31:01 - loss: 1.6650 - acc: 0.6068
...
Training time(secs): 95846.50487065315
Max training iterations: 100
Training time / Max training iterations: 958.4650487065315Simple CNN model acc: 60.77%
Saved model to disk
Done building estimator
Dummy y pred dev class 151
```
#### Modelv2
This had been running for ~30 hours
Epoch 1/100
2018-12-05 06:50:44.482968:
Stopped at December 6, 11:39am.
```
Epoch 73/100
4800000/4800000 [==============================] - 553s 115us/step - loss: 1.9147 - acc: 0.5562
Epoch 74/100
4800000/4800000 [==============================] - 550s 115us/step - loss: 1.9151 - acc: 0.5563
Epoch 75/100
4800000/4800000 [==============================] - 554s 115us/step - loss: 1.9157 - acc: 0.5560
Epoch 76/100
4800000/4800000 [==============================] - 560s 117us/step - loss: 1.9139 - acc: 0.5564
Epoch 77/100
4800000/4800000 [==============================] - 3005s 626us/step - loss: 1.9141 - acc: 0.5565
Epoch 78/100
4800000/4800000 [==============================] - 32914s 7ms/step - loss: 1.9141 - acc: 0.5563
Epoch 79/100
2603000/4800000 [===============>..............] - ETA: 12:48:37 - loss: 1.9101 - acc: 0.5573

```


## Models Version

### CNN Model v1
* This is the first model and it used 2 convolutions.

### CNN Model v2
* This is the second iteration of the model and we reduce number of convolutions to 1 to see what the impact would be.

### CNN Model v3
* This is the third iteration of the model. It has only 1 convolution layer and 1 fully connected layer (removed max pooling layer).

### CNN Model v4
* This is the fourth iteration of the model and we used v2 but had a dense layer of 256 (instead of 128).


## Transfer learning test run
Test with 2 epochs over night.
```
Tuning our last custom layer...
Epoch 1/2
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2041 thread 5 bound to OS proc set 5
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2252 thread 10 bound to OS proc set 0
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2253 thread 11 bound to OS proc set 1
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2255 thread 13 bound to OS proc set 3
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2256 thread 14 bound to OS proc set 4
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2251 thread 9 bound to OS proc set 9
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2249 thread 7 bound to OS proc set 7
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2254 thread 12 bound to OS proc set 2
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2248 thread 6 bound to OS proc set 6
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2250 thread 8 bound to OS proc set 8
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2040 thread 15 bound to OS proc set 5
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2257 thread 16 bound to OS proc set 6
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2259 thread 18 bound to OS proc set 8
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2258 thread 17 bound to OS proc set 7
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2260 thread 19 bound to OS proc set 9
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2261 thread 20 bound to OS proc set 0
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2262 thread 21 bound to OS proc set 1
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2263 thread 22 bound to OS proc set 2
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2264 thread 23 bound to OS proc set 3
OMP: Info #250: KMP_AFFINITY: pid 2023 tid 2265 thread 24 bound to OS proc set 4
48000/48000 [==============================] - 6553s 137ms/step - loss: 0.3723 - acc: 0.8586
Epoch 2/2
48000/48000 [==============================] - 8123s 169ms/step - loss: 0.2246 - acc: 0.9137

Tuning the last 2 inceptions layers ...
Epoch 1/2
48000/48000 [==============================] - 4809s 100ms/step - loss: 0.1893 - acc: 0.9301
Epoch 2/2
48000/48000 [==============================] - 2187s 46ms/step - loss: 0.1820 - acc: 0.9325
('Model metric names: ', ['loss', 'acc'])
-------------------

Training time(secs): 22645.9083531
Max training iterations: 2
Training time / Max training iterations: 11322.9541765Simple CNN model (tranfer learning) acc: 46.47%

```

###Repeat of run but with GPUs
We see a 25x speed up!
```
(310, 'mixed10')
Tuning the last 2 inceptions layers ...
Epoch 1/2
48000/48000 [==============================] - 145s 3ms/step - loss: 0.2210 - acc: 0.9168
Epoch 2/2
48000/48000 [==============================] - 133s 3ms/step - loss: 0.2062 - acc: 0.9226
('Model metric names: ', ['loss', 'acc'])
-------------------

Training time(secs): 887.143157005
Max training iterations: 2
```



Inception with 20 epochs of training.
```
Epoch 12/20
48000/48000 [==============================] - 132s 3ms/step - loss: 0.0610 - acc: 0.9832
Epoch 13/20
48000/48000 [==============================] - 132s 3ms/step - loss: 0.0598 - acc: 0.9839
Epoch 14/20
48000/48000 [==============================] - 133s 3ms/step - loss: 0.0585 - acc: 0.9845
Epoch 15/20
48000/48000 [==============================] - 132s 3ms/step - loss: 0.0572 - acc: 0.9850
Epoch 16/20
48000/48000 [==============================] - 133s 3ms/step - loss: 0.0553 - acc: 0.9861
Epoch 17/20
48000/48000 [==============================] - 132s 3ms/step - loss: 0.0547 - acc: 0.9859
Epoch 18/20
48000/48000 [==============================] - 132s 3ms/step - loss: 0.0537 - acc: 0.9866
Epoch 19/20
48000/48000 [==============================] - 133s 3ms/step - loss: 0.0531 - acc: 0.9862
Epoch 20/20
48000/48000 [==============================] - 134s 3ms/step - loss: 0.0519 - acc: 0.9874
('Model metric names: ', ['loss', 'acc'])
-------------------

Training time(secs): 5493.44327402
Max training iterations: 20
Training time / Max training iterations: 274.672163701Tranfer learning model result  acc: 48.77%
```

###ResNet50 with 5 max iterations

```
(166, 'bn5c_branch2a')
(167, 'activation_141')
(168, 'res5c_branch2b')
(169, 'bn5c_branch2b')
(170, 'activation_142')
(171, 'res5c_branch2c')
(172, 'bn5c_branch2c')
(173, 'add_16')
(174, 'activation_143')
('Model metric names: ', ['loss', 'acc'])
-------------------

Training time(secs): 1012.68207002
Max training iterations: 5
Training time / Max training iterations: 202.536414003Tranfer learning model result  acc: 51.00%
```

MobileNet Transfer Learning, 3 classes
```
-------------------
Using model: MobileNet

Training time(secs): 10289.715752363205
Max training iterations: 100
Training time / Max training iterations: 102.89715752363205Tranfer learning model result  acc: 75.40%
```

### CNN Model v4 with binaziration
This performed very poorly. Not sure if there is an issue with the setup. Got 2% accuracy which is
just about random. 

```
800000/800000 [==============================] - 11s 13us/step - loss: 15.7957 - acc: 0.0200 - val_loss: 15.7957 - val_acc: 0.0200
Epoch 47/100
800000/800000 [==============================] - 11s 13us/step - loss: 15.7957 - acc: 0.0200 - val_loss: 15.7957 - val_acc: 0.0200
Epoch 48/100
800000/800000 [==============================] - 11s 13us/step - loss: 15.7957 - acc: 0.0200 - val_loss: 15.7957 - val_acc: 0.0200
Epoch 49/100
800000/800000 [==============================] - 10s 13us/step - loss: 15.7957 - acc: 0.0200 - val_loss: 15.7957 - val_acc: 0.0200
Epoch 50/100
800000/800000 [==============================] - 11s 13us/step - loss: 15.7957 - acc: 0.0200 - val_loss: 15.7957 - val_acc: 0.0200
Epoch 51/100
641500/800000 [=======================>......] - ETA: 2s - loss: 15.7981 - acc: 0.0199^CTraceback (most recent call last):
```

## TensorBoard
For CNN training at least, we have integrated tensor board support. You can view the training logs as follows

```
$ tensorboard --logdir=results/tensorboard
```

Note that because the log files can be very large, we do not check these into our version control. These logs
files will have on the machine(s) that ran the experiments. In this case, this is the google cloud vm. 

### Inception stuck
Been running for 15h. Could be to lakc of tuning last layers; oalso the google VM was being really slow. May have been on a bad host.
Systrae was showing it stuck on sched_yield() which emant the threads were effectively refusing to make progress and do useful work and
thus procastinating

```
Convert data to shape: (299, 299)
Convert data to shape: (299, 299)
Done preprocessing dataset
Building transfer learning model...
Tuning our last custom layer...
Epoch 1/100
48000/48000 [==============================] - 3444s 72ms/step - loss: 0.3918 - acc: 0.8406
Epoch 2/100
48000/48000 [==============================] - 4733s 99ms/step - loss: 0.2413 - acc: 0.9087
Epoch 3/100
48000/48000 [==============================] - 6048s 126ms/step - loss: 0.2327 - acc: 0.9118
Epoch 4/100
48000/48000 [==============================] - 6082s 127ms/step - loss: 0.2114 - acc: 0.9199
Epoch 5/100
48000/48000 [==============================] - 6029s 126ms/step - loss: 0.2078 - acc: 0.9214
Epoch 6/100
48000/48000 [==============================] - 6077s 127ms/step - loss: 0.2053 - acc: 0.9205
Epoch 7/100
48000/48000 [==============================] - 6042s 126ms/step - loss: 0.1929 - acc: 0.9266
Epoch 8/100
48000/48000 [==============================] - 6080s 127ms/step - loss: 0.1886 - acc: 0.9270
Epoch 9/100
41000/48000 [========================>.....] - ETA: 13:46 - loss: 0.1857 - acc: 0.9282
```

## CNN Network Model Architecture
CNN network with 2 convolution layers.
![CNN_network](https://github.com/jervisfm/cs229-project/blob/master/cnn-model-network-architecture.png)


###Inception  transfer learning repeat, 200 epochs total
--max_iteration=100 but we tune both source model and our own custom layer so overall, had 200 epochs total.

```
Epoch 99/100
48000/48000 [==============================] - 120s 2ms/step - loss: 4.8579e-04 - acc: 0.9999
Epoch 100/100
48000/48000 [==============================] - 120s 3ms/step - loss: 3.6866e-04 - acc: 1.0000
('Model metric names: ', ['loss', 'acc'])
Training time(secs): 23094.9305902
Max training iterations: 100
Training time / Max training iterations: 230.949305902Tranfer learning model result  acc: 41.67%
```


### Inception transfer learning repeat, 100 epochs total
--max_iteration=50, but we tune both source moudel and custom layer so overall is 100 epochs total.
nax accuracy 45%.
```
Epoch 43/50
48000/48000 [==============================] - 225s 5ms/step - loss: 0.0029 - acc: 1.0000 - val_loss: 8.6998 - val_acc: 0.4565
Epoch 44/50
48000/48000 [==============================] - 241s 5ms/step - loss: 0.0030 - acc: 0.9999 - val_loss: 8.6987 - val_acc: 0.4570
Epoch 45/50
48000/48000 [==============================] - 249s 5ms/step - loss: 0.0029 - acc: 1.0000 - val_loss: 8.6974 - val_acc: 0.4572
Epoch 46/50
48000/48000 [==============================] - 232s 5ms/step - loss: 0.0028 - acc: 0.9999 - val_loss: 8.6937 - val_acc: 0.4573
Epoch 47/50
48000/48000 [==============================] - 402s 8ms/step - loss: 0.0029 - acc: 1.0000 - val_loss: 8.6995 - val_acc: 0.4567
Epoch 48/50
48000/48000 [==============================] - 240s 5ms/step - loss: 0.0028 - acc: 1.0000 - val_loss: 8.7022 - val_acc: 0.4560
Epoch 49/50
48000/48000 [==============================] - 246s 5ms/step - loss: 0.0027 - acc: 1.0000 - val_loss: 8.6982 - val_acc: 0.4568
Epoch 50/50
48000/48000 [==============================] - 244s 5ms/step - loss: 0.0028 - acc: 0.9999 - val_loss: 8.6989 - val_acc: 0.4572
('Model metric names: ', ['loss', 'acc'])
-------------------

Training time(secs): 27001.143337
Max training iterations: 50
Training time / Max training iterations: 540.02286674Tranfer learning model result  acc: 45.72%
cs229@cs229-gpu-vm:~/cs229-project$
```

#### VGG
```

48000/48000 [==============================] - 201s 4ms/step - loss: 0.0285 - acc: 0.9901
Epoch 18/20
48000/48000 [==============================] - 201s 4ms/step - loss: 0.0286 - acc: 0.9903
Epoch 19/20
48000/48000 [==============================] - 201s 4ms/step - loss: 0.0261 - acc: 0.9913
Epoch 20/20
48000/48000 [==============================] - 201s 4ms/step - loss: 0.0227 - acc: 0.9927
0 input_2
1 block1_conv1
2 block1_conv2
3 block1_pool
4 block2_conv1
5 block2_conv2
6 block2_pool
7 block3_conv1
8 block3_conv2
9 block3_conv3
10 block3_conv4
11 block3_pool
12 block4_conv1
13 block4_conv2
14 block4_conv3
15 block4_conv4
16 block4_pool
17 block5_conv1
18 block5_conv2
19 block5_conv3
20 block5_conv4
21 block5_pool
Model metric names:  ['loss', 'acc']
-------------------
Using model: VGG19

Training time(secs): 4140.123324394226
Max training iterations: 20
Training time / Max training iterations: 207.0061662197113Tranfer learning model result  acc: 95.58%
```


##### VGG16 test score result

```
Got accuracy of 95%
12 block4_conv1
13 block4_conv2
14 block4_conv3
15 block4_conv4
16 block4_pool
17 block5_conv1
18 block5_conv2
19 block5_conv3
20 block5_conv4
21 block5_pool
Model metric names:  ['loss', 'acc']
-------------------
Using model: VGG19

Training time(secs): 5546.821496963501
Max training iterations: 20
Training time / Max training iterations: 277.341074848175Tranfer learning model result DEV  acc: 95.57%Tranfer learning model result TEST  acc: 95.57%
```