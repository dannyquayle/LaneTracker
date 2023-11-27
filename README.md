# README

## Introduction

This method use a fine-tuned CLRNet model as the detector, but users can change it to the detector they use.



## Dependencies:

- Python >= 3.8 (tested with Python3.8)
- PyTorch >= 1.6 (tested with Pytorch1.6)
- CUDA (tested with cuda10.2)
- Other dependencies described in requirements.txt

To install required dependencies run:
```
$ pip install -r requirements.txt
```



## Demo:

To run the tracker with the provided detections:

```
$ cd path/to/project
$ python lanetracker.py
```

And the results of assigning instance IDs are saved in `output`

To show the visible results you need to:

1. Use your own private dataset. Or contact us and we will send you the download address of our own dataset.

2. Create a symbolic link to the dataset
  ```
  $ ln -s /path/to/dataset mot_benchmark
  ```
3. Run the demo with the ```--display``` flag
  ```
  $ python lanetracker.py --display
  ```

To calculate lane offset and show the result:

```
$ python demo/calc_offset.py
```

The trajectory of the vehicle will be saved in `demo/lane_offset_plot.png`. It's corresponding visible result on our private dataset compared with the counterpart of the baseline method shows in `demo/demo.avi`.