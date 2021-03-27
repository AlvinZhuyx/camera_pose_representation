Here is the code for our experiments on Gibson Room Dataset:

This dataset contains the modeling of both 2D position change in polar coordinate system and 1D orientation change as 
an individual orientation vector.

1. To run our code, please first download the dataset. 
We use the dataset provided by the generative query network, following is their link:
https://github.com/deepmind/gqn-datasets

Please download the dataset room_free_camera_no_object_rotation/test to the dataset/gqn_dataset foler (after doing that, you should
get path like /dataset/gqn-datasets/room_free_camera_no_object_rotation/test with contains lots of tf_record files)

Then please run the code:
python tfrecord-converter.py 
(we get this code from https://github.com/wohlert/generative-query-network-pytorch/tree/master/scripts, it extract the compress tf record files into several small files with allows to be used by torch and also allow us to go over all of them in the testing phase)
The code will generate another folder /dataset/gqn-datasets/torch which contains a batch of files like "001-of-240-00.pt.gz".

2. For novel view synthesis experiment, get into the "code" directory and run "novel_view_synthesis.py"\
* To reproduce the test the robustness to noise result: 
Please first download our pretrained checkpoints:
https://drive.google.com/file/d/1iOpNPvPwUmDJFv568oXPslLyRlYqf7Wg/view?usp=sharing
Please extract it to the code directory (You should get the folder gqn_room/code/checkpoint)
Then simply run:
python novel_view_synthesis.py

To retrain the model from scratch, use the command:  
python novel_view_synthesis.py --train True

For our experiment, we run the code at a single Titan RTX GPU(with 24GB memory) for 4 days. 

3. For camera pose estimation experiment, get into the "code" directory and run "inference.py"
Note that you need to first run the training to get the pose representation (or download our checkpoint), then you can run this code.     
To reproduce our inference result, please first down load our pretrained inference model checkpoint:
https://drive.google.com/file/d/1K1ASiBB4t7cOmFmNn1ZgnQSwLC353jeq/view?usp=sharing
Please extract to the code directory (You should get the folder gqn_room/code/checkpoint_infer)
(also please download the novel view synthesis checkpoint follows 2)
Then simply run:
python inference.py

To retrain the model from scratch, use the command: python inference.py --train True
