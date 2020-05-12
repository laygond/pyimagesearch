
## Source 
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

## Important Additional Notes
#### Tensorflow version below 1.5 and MobileNet
If you use TF version below 1.5 you will need this temporary fix to solve the mobilenet save load issue in `train_mask_detector.py` as stated (here)[https://github.com/tensorflow/tensorflow/issues/22697]
This is fixed with TF version 1.15 and in TF 2. 

Open editor at tensorflow/python/keras/layers/advanced_activations.py and (around line 310), after the super() call in the ReLu init() function, add the following lines of code:   
```
 if type(max_value) is dict:
        max_value = max_value['value']
if type(negative_slope) is dict:
    negative_slope = negative_slope['value']
if type(threshold) is dict:
    threshold = threshold['value']
```
I also replaced the default model path from `mask_detector.model` to `mask_detector.h5` in `train_mask_detector.py` and in the two `detect_mask` files: image and video
```
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.h5", # model",
	help="path to output face mask detector model")
```

#### Numpy version above 1.16.4
Newer versions of Tensorflow can cause compatibility warning messages with numpy versions. if you get them you can revert to numpy version 1.16.4 as stated (here)[https://github.com/tensorflow/tensorflow/issues/30427]
```
pip install numpy==1.16.4
```

#### TF 2.0 Metrics [acc] vs [accuracy]
Stick with \[acc\] for backward compatibility. As stated by [codesofinternet](https://www.codesofinterest.com/2020/01/fixing-keyerror-acc-valacc-keras.html) there are some breaking changes

In `train_mask_detector.py` replace "accuracy" with "acc"
```
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
```

You can read the official release notes (here)[https://github.com/keras-team/keras/releases/tag/2.3.0]. What this means is that if you specify metrics=["accuracy"] in the model.compile(), then the history object will have the keys as 'accuracy' and 'val_accuracy'. While if you specify it as metrics=["acc"] then they will be reported with the keys 'acc' and 'val_acc'.
