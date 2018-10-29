# cs229-project.

## Team Members
- Connie 
- Minh
- Jervis


## Introduction
Goal is to train a classifier that can recognize / classify handwritten digits. 

Dataset to use is the Google Quick Draw dataset.

## Dowonloading Dataset.

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

## Initial Baseline
1 epoch, batch_size=500

```
Epoch 1/1
302140/302140 [==============================] - 6s 21us/step - loss: 0.8110 - acc: 0.5804
302141/302141 [==============================] - 4s 15us/step
Epoch 1/1
302141/302141 [==============================] - 6s 21us/step - loss: 1.0513 - acc: 0.5080
302140/302140 [==============================] - 5s 15us/step
Baseline: 60.44% (9.49%)
```