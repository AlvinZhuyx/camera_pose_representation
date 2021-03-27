Here is the code for our experiments on Gibson Room Dataset:

This dataset contains the modeling of both 2D position change in polar coordinate system and 1D orientation change as 
an individual orientation vector.

1. To run our code, please first download our dataset. 
We put our dataset on the google drive: https://drive.google.com/file/d/1td2ej_5g_Ab784hK1sg6-liEgNWSaXpw/view?usp=sharing
Please download this dataset, and extract the dataset under the dataset directory.
After extraction, you should get the "dataset" directory contains a directory called "gibson_room" and the "gibson_room" 
directory contains 12 different scene driectories ("Angilola" "Annawan" "Beach" "Brentsville" "Caruthers" "Convoy" "Cooperstown"
"Denmark" "Dunmor" "Elmira" "Espanola" "Kerrtown"). Each of them contains images and poses in a couple of square areas where our agent moves in

2. For novel view synthesis experiment, get into the "code" directory and run "novel_view_synthesis.py"
* To reproduce the test the robustness to noise result: 
Please first download our pretrained checkpoints:
https://drive.google.com/file/d/1QU_eMV6eaadSe_p1o8MGTiCB3UVshMyO/view?usp=sharing
Please extract it to the code directory (You should get the folder gibson_room/code/checkpoint)
Then simply run:
python novel_view_synthesis.py

To retrain the model from scratch, use the command:  
python novel_view_synthesis.py --train True

For our experiment, we run the code at a single Titan RTX GPU(with 24GB memory) for 4 days. 

3. For camera pose estimation experiment, get into the "code" directory and run "inference.py"
Note that you need to first run the training to get the pose representation (or download our checkpoint), then you can run this code.     
To reproduce our inference result, please first down load our pretrained inference model checkpoint:
https://drive.google.com/file/d/16OiIY3wem0pDuNDJxBhKXr3aWei1s5KB/view?usp=sharing
Please extract to the code directory (You should get the folder gibson_room/code/checkpoint_infer)
(also please download the novel view synthesis checkpoint follows 2)
Then simply run:
python inference.py

To retrain the model from scratch, use the command: python inference.py --train True
