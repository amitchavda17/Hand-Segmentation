# Hand Segmentation and Action Classification using U-Net
make a virtual enviornment and install the requirements
```
pip install -r requirements.txt
```
A) Training Hand Segmentation Model 

 We use online available dataset for training the segmentation model:
    
    a. EYTH (EgoYoutubeHands)
    b. HOF(HandOverFace)
    
```Hand_Seg_Unet.ipynb```[clickhere](Hand_Seg_Unet.ipynb)


B) Running Segmentation Inference 

Inference on Images
```
python images_inference.py --source [path to image folder] --output [path to results base folder (default results)] 
```

Inference on Videos

```
python video_inference.py --input_path [path to video] --output_path [path to results base folder (default output_video)]  --show_video [1 is you want to see live inference (default 0)]
```
Eg.
```
python video_inference.py --input_path ./video_2.mp4 
```
**output_video** folder has infernece outputs


C) Action Classification 
    
Problem :Classifiying action in the cooking video as stirring or adding ingrident

Solution: We create a Image classification model that takes the segmented hand image as input and classifies the image as either adding ingrident, stirring or background 

```action_classification.ipynb``` [clickhere](action_classification.ipynb)

D) Action Classification Inference

End to end Inferencing with hand segmentation and action classification

```
python video_action_inference.py --input_path [path to video file] --output_path [path to results base folder (default output_video_action)]  --show_video [1 is you want to see live inference (default 0)]
```
**output_video_action** folder has infernece outputs