# Jetbot autonomous driving project

### Overview

This project aimed at building an autonomous driving system for a [**Jetbot**](https://jetbot.org/master/) using deep learning techniques. The system is designed to navigate through simple car tracks by processing camera input and making real-time driving decisions.

![Jetbots](assets/jetbot_on_track.jpg)

**Figure 1:** jetbots standing in line. Ready to take the most important race of their life, that will decide the robotics2 grades of tens of AI students. 

![Camera Input](assets/camera_input.png)

**Figure 2:** Example input from the Jetbot's camera

### Data Collection

To train our model, we collected data by manually driving the Jetbot around the track while recording the camera feed and corresponding steering commands. This dataset was then used to train a our first approach, a convolutional neural network (CNN) that takes the camera input and predicts the steering angle.

For our second approach, to-point-driving, we manually annotated a dataset with target points on the track and trained a model to predict the next target point based on the current camera input. For annotation we used script available in `src/data_methods/data_processing/to_point_driving/label_images_custom.ipynb`.

## Data preprocessing

### All models

In both approaches, we first filtered out images that were not useful for training, such as those where jetbot stayed out of track or where jetbot was standing still.

Also during training, all models were additionally augmented by random color jittering to ensure better generalization and robustness to different lighting conditions. This was done using `torchvision.transforms.ColorJitter` with parameters `brightness=0.2`, `contrast=0.2`, `saturation=0.2`, and `hue=0.1`.

### Imitation driving

Our preprocessing pipeline for the imitation driving approach includes the following steps:
- in `csv_preprocess.py` we concate all the csv files with controler inputs and shift them by 1 frame, as original 10fps recording was pairing frames with old controller inputs, which would make training impossible.
- in `images_preprocess.py` we place all the images in a single directory and ensure their names are consistent with csv (they have to have the same number of digits in decimal part for efficient pairing csv row <-> image, but by default they don't)
- in `sync_images_csv.py` we remove images without matching row in csv and remove rows in csv without an image.
- Finally, in `data_inspection_steering.ipynb` we mirror each image with respect to y axis and flip its steering value. This way we have 2x larger dataset 2*1948=3896 images with controler data, which enhances the model's ability to generalize and handle various driving scenarios.

### To-point-driving

Well, as points were annotated manually, there is no additional scripts. all of the data is already preprocessed and ready for training. The only need was to manually remove images where jetbot was out of track or standing still, but that was done during annotation process, so no additional scripts were needed for that.

## Model Architecture

### to-point driving

Our CNN architecture for the to-point driving approach was directly taken from the [NVIDIA Jetbot repository tutorial on road_following](https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/road_following).

The model consists of a pretrained backbone network (originally ResNet18, but due to high latency we switched to MobileNetV3) followed by a fully connected layer that outputs the x and y coordinates of the target point. The model was trained using mean squared error loss between the predicted and actual target points.

### Imitation driving

For the imitation driving approach, we used a similar CNN architecture, but instead of predicting target points, the output layer was designed to predict the steering angle directly, in range [-1, 1]. It  model uses MobileNetV4 Conv Small (pretrained on ImageNet), with the final classifier replaced by a custom head: Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→2) → Tanh.

The model was trained using mean squared error loss between the predicted steering angle and the actual steering angle from the dataset.

## Training and Evaluation

Both models were trained using the Adam optimizer with a learning rate of 0.001/0.0005 and a batch size of 32. We trained for 50 epochs, monitoring the training and validation loss to ensure that the model was learning effectively without overfitting. To ensure proper training, we split our dataset into training and validation sets, using an 80-20 split.

![imitation driving losses](assets/imitation_train_val_loss.png)

**Figure 4:** We can see that loss during training is decreasing, which indicates that the model is learning to predict steering angles based on the camera input. The validation loss is suprisingly staying almost the same, which may indicate that the model is not generalizing well to unseen data. However, manual checks found model predictions to be reasonable, so we decided to proceed with deployment on Jetbot and evaluate its performance in real-world conditions.

![point-to driving losses](assets/point_to_train_val_loss.png)

**Figure 5:** Here both training and validation losses are decreasing, which indicates that the model is learning to predict target points based on the camera input.

Before testing it on Jetbot, we evaluated the performance of both models by taking sample images from the validation set and comparing the predicted steering angles or target points with the actual values. This allowed us to verify that the models were learning to make reasonable predictions based on the camera input.

![Model Prediction (point to driving)](assets/point_to_model_prediction.png)

Figure 5: We can see that the predicted target point (red dot) is reasonably close to the actual target point (green dot), indicating that the model has learned to predict target points based on the camera input.

## Transformation to steering commands

- For the imitation driving model, the output is already a steering angle, so we simply need to convert it to a format that can be sent to the Jetbot's motor controller. This involves scaling the predicted steering angle to the appropriate range and sending it as a command to the motors.
- For the to-point driving model, we closely follow the approach described in the NVIDIA Jetbot repository tutorial on road_following. We calculate the angle between the current position of the Jetbot and the predicted target point, and then convert that angle into a steering command that can be sent to the motors. This involves some trigonometric calculations to determine the appropriate steering angle based on the relative position of the target point. We also tried some smoothing techniques to reduce the jitteriness of the steering commands, yet it caused to increasy latency even more, so we had to remove it in the end.

## Deployment on Jetbot

After training, we tried to deploy both models on jetbot. Sadly, due to many technical difficulties (see section "Issues encountered") we needed to convert all models to ONNX format and run them using ONNX Runtime. This conversion process was straightforward for the imitation driving model, but for the to-point driving model we had to make some adjustments to the architecture to ensure compatibility with ONNX. Sadly, even after conversion, the to-point driving still faced some issues with latency, which made it too buggy on the track to work. Luckily, the imitation driving model worked well and was able to navigate the track autonomously, demonstrating the effectiveness of our approach. 

It was able to achieve **1 lap in 18 seconds**, which exactly matched our score from manual driving. However we belive that 

- [link to video of our jetbot **driving autonomously** (with live comments of team members!)](https://drive.google.com/file/d/1Pb10zPwQQIwnaZj92rRBH_AwpXqcleDk/view?usp=sharing)


## Issues encountered

### Lack of GPU availability

Despite jetbot formally having a GPU, we were not able to utilize it for our models due to strange deployement of our notebooks instance on jetbot. As we tried to repair it, we ended up removing pytorch from jetbot, causing a lot more issues. Finally GPU was available, yet required very old version of pytorch.

### Pytorch installation issues

After removing pytorch, we had to reinstall it, but due to some dependency issues, we were not able to install the correct version of pytorch that would be compatible with our models. We somehow managed to find a version that worked, but it was very old one (1.3.0), which caused some issues with our code and forced to downgrade onnx opset_version from 17 to 11. Due to that, we also needed to monkey patch some model layers (like HardSigmoid) to make them compatible with older onnx opset.

Suprisingy though it gave as access to CUDA, which we were not able to use before. Sadly, it didn't improve latency so much (only by about 10ms on imitation driving model), which made us think that the main bottleneck was not in the GPU, but rather in the CPU or in the data transfer between CPU and GPU.

### Inportability to other robots

We also tried to change robot on one labalatories, but we encountered problems:
- on one robot, it was impossible to install necessary dependency (ONNX runtime), which made it impossible to run our models on that robot.
- on other robot, imports both in the notebook and python script refused to cooperate due to numpy problems, but removing numpy caused problems in it's reinstallation, which made it impossible to run our code on those robots as well.

### Oversmoothing of steering commands
In an attempt to reduce the jitteriness of the steering commands, we implemented a smoothing techniques that averaged the predicted steering angles over a short window of time. However, it introduced even more latency to the system. We also had problem that trying to fix latency we implemented multiple smoothing layers on top of each other (in 1 moment we had 4 layers of smoothing), which caused the steering commands to become too sluggish and unresponsive, especially in tight turns. Ultimately, we had to remove all smoothing layers to achieve better performance.

### Latency issues
Even after optimizing our models and converting them to ONNX format, we still faced significant latency issues, especially with the to-point driving model. This was likely due to the complexity of the model and the limitations of the hardware on the Jetbot. Despite our efforts to optimize the model architecture and reduce latency, we were not able to achieve fair performance with the to-point driving approach, which ultimately led us to focus on the imitation driving model for deployment.

| Model    | CPU             |
| -------- | -------         | 
| Imitation | 70ms (10ms)    |
| To-point  | 100ms (150ms)  |

**Figure 7:** Latency measurements for both models on CPU and GPU. The numbers in parentheses represent standard deviations. The to-point driving model was not able to run on GPU due to compatibility issues with the older version of PyTorch (there were some problems with torchvsion and onnx conversion).



## Error analysis
After testing our models on the track, we observed that the imitation driving model performed reasonably well, but there were still some instances where it struggled to navigate  turns or recover from being slightly off the track. It may be due to: 
- speed-quality tradeoff, as we had increase speed to achieve reasonable time per lap, which created more risk of too late reaction to turns and obstacles on the track.
- lack of any smoothing, causing model to turn too sharply in some cases. 
- Our dataset not being good enough, as we only collected data from a few laps around the track, which were created by a single driver (one of us), which may have introduced some bias into the model's learning process. Additionally, our driving style was quite sharp and risky, which may have caused the model to learn more aggressive steering behaviors that are not ideal for smooth navigation.



## runnig the code.
The commends needed to fully reproduce the results — from data collection to model deployment can be found in README.md files in folders `data_methods`, `imitation_driving` and  `to_point_driving`.

In order to just run the models run:
- for Imitation driving - `python3 sakupen_circles/src/imitation_driving/driving/steering_driver.py`
- for To-point driving - `python3 sakupen_circles/src/to_point_driving/driving/autonomous_drive.py`
    - If the version of the offset in onnx model is different than required — use `sakupen_circles/src/to_point_driving/helpers/export_onnx.py` to export model to corret version

all scripts must be executed on the Jetbot with the camera connected and the model files present under sakupen_circles/models/


## Conclusions

The imitation-driving approach proved to be the more practical solution, matching manual-driving lap times despite the Jetbot's hardware constraints, while to-point-driving remained promising but was held back by latency and compatibility issues rather than a flawed concept. Most challenges came from infrastructure rather than the models themselves, with the broken PyTorch install and forced opset downgrade costing significant time. Future work should focus on collecting a larger, more varied dataset, smoothing steering output, adjusting the robots controls based on models output, and better balancing speed against control precision to improve turn handling and recovery.


## repository

a full codebase of the project can be found here:
- [https://github.com/noNScop/Jetbot](https://github.com/noNScop/Jetbot )



## Authors
- Oliwier Necelman
- Karol Sroka
- Igor Szymczak

## Team name:
![our team](assets/sakupen_circles.jpeg)

**Figure 9:** A thumbnail from the Geometry Dash level that inspired our team's name — **"Sakupen Circles"**.

A full video of this level an be found here: 
[youtube video](https://www.youtube.com/watch?v=b66_Oe4Xo1w).



