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

Then you can ssh into the VM with:

```
$ gcloud compute ssh --project cs229-2018 --zone "us-west1-b" cs229@cs229-vm-vm
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
Prediction: ['eraser' 'eraser' 'eraser' ... 'banana' 'banana' 'banana']
Actual Label: ['banana' 'banana' 'banana' ... 'belt' 'belt' 'belt']
Acurracy: 0.414380952381
Training time(secs): 199.068641186
('Labels: ', ['banana', 'eraser', 'belt'])
('Confusion matrix: ', array([[3914, 2580,  506],
       [3545, 2688,  767],
       [3612, 1288, 2100]]))
Confusion matrix, without normalization
[[3914 2580  506]
 [3545 2688  767]
 [3612 1288 2100]]
Normalized confusion matrix
[[0.56 0.37 0.07]
 [0.51 0.38 0.11]
 [0.52 0.18 0.3 ]]
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
