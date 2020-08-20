## Source
https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/

## Important Additional Notes
Yolo V3 was trained on the coco dataset and the weights can be downloaded from terminal like
```
wget https://pjreddie.com/media/files/yolov3.weights
```
Once downloaded, placed them under `social-distance-detector/yolo-coco/`

To run you must execute this in terminal
```
python social_distance_detector.py --input pedestrians.mp4
```