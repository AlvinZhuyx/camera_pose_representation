Here is the instruction for the 7Scene dataset.
Our camera pose estimation code is adapted from the offical code of "Geometry-Aware Learning of Maps for Camera Localization" (Their origin github link: https://github.com/NVlabs/geomapnet)
Note that our experiments are only based on PoseNet so we only modify the branch for PoseNet.

To reproduce our result reported in our paper, please follow the following instruction:
1. Download the dataset:
Here is the dataset containing the pretrained embedding of each image:
(we stored our camera pose embedding for each image in 'frame_xxxxxx_posevec.npy'. You can also download the original 7Scene and encode the embedding your self, please see the part of encoding dataset)  
https://drive.google.com/file/d/1TrxjYQyG0NoZQoXfZFd1Ug1UqKN0vU6l/view?usp=sharing
Please download this dataset and extract the contains to the dataset folder (You should get "dataset/7Scenes")

2. Download our pretrained checkpoints:
Here is our pretrained checkpoints:
https://drive.google.com/file/d/1lfiQ60jiK_F14CuFnrwk0LvZS9fACiT3/view?usp=sharing
Please download this embedding to the folder "code_camera_pose_estimation/scripts". (You should get "code_camera_pose_estimation/scripts/logs_and_checkpoints" and folder contains 7 folders for the 7 datasets)

3. Run the test:
Please using the following command(suppose you are under our 7scenes directory):
cd ./code_camera_pose_estimation/scripts
chmod +x test_all.sh
./test_all.sh which will show the test result on the 7 scenes

4. Retrain the camera pose estimation model(suppose you are under our 7scenes directory):
cd ./code_camera_pose_estimation/scripts
chmod +x train_all.sh
./train_all.sh which will show the test result on the 7 scenes

5. Encode the embedding:
To encode the embedding yourself, please download our pretrained checkpoint from novel_view_synthesis:
https://drive.google.com/file/d/19xilGqKWyyxu08Zn7np7aJNQoPfWRN3u/view?usp=sharing
extract to folder './code_novel_view' (you should get './code_novel_view/checkpoint')
Also download the origin 7Scene dataset and put to the dataset folder
then please run(suppose you are under our 7scenes directory)):

cd ./code_novel_view
python compress_dataset.py
python encode_dataset.python

6.Retrain the embedding through novel_view_synthesis:
You can also retrain the camera pose embedding through novel_view_synthesis. (Note that due to randomness, retrain the embedding may give you a slightly different embedding and you may need to tune the downstream hyperparameters to adjust for that)
Please first make sure you download the dataset (either from our provided link or from the original dataset)
Also please first run the compress_dataset.py, this will provide the intermediate file using by our dataloader.
Then simply run:
cd ./code_novel_view
python main.py 







  
 