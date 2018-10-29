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

