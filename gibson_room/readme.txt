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

* For training the model, use the command: python novel_view_synthesis.py --train True
* After training, test the reconstruction and test the robustness to noise: python novel_view_synthesis.py

For our experiment, we run the code at a single Titan RTX GPU(with 24GB memory) for 4 days. 

3. For camera pose estimation experiment, get into the "code" directory and run "inference.py"
Note that you need to first run the training to get the pose representation, then you can run this code.     
* For training the model, use the command: python inference.py --train True
* For testing: python inference.py