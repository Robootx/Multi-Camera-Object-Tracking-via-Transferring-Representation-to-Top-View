# Multi Camera Object Tracking via Deep Metric Learning
Transferring Representation to ‘Top View’ based on deep metric learning
![image](pca_visualize/top_view_net.png?raw=true "Optional Title") 
![image](pca_visualize/5_0.png?raw=true "Optional Title") 
![image](pca_visualize/5_1.png?raw=true "Optional Title") 
![image](pca_visualize/5_2.png?raw=true "Optional Title") 
![image](pca_visualize/5_3.png?raw=true "Optional Title") 
![image](pca_visualize/5_4.png?raw=true "Optional Title") 
 

**Visualization of 'top view' by applying PCA on learned embeddings.**

[demo video on EPFL dataset](https://www.youtube.com/watch?v=sroLfpX4F0w)

For inference:

Download data and trained model from [data](https://drive.google.com/file/d/1Io3nNM2kjJ08GSC2vqLqL2XezvFBlsKo/view?usp=sharing)
and [trained model](https://drive.google.com/file/d/1RgKUQt55CChsN0lX2JTxMdWBpz5qZ7If/view?usp=sharing), put them in ../box2vec/data and ../box2vec/model directory respectively

Download test videos "4 people indoor sequence"(4p-c0.avi, 4p-c1.avi, 4p-c2.avi, 4p-c3.avi) from [“EPFL” data set: Multi-camera Pedestrian Videos
](https://cvlab.epfl.ch/data/data-pom-index-php/) and detection files (4p-c0.pickle, 4p-c1.pickle, 4p-c2.pickle,4p-c3.pickle) from [detection file for demo](https://drive.google.com/file/d/12MWB_CMOdDwfeG_ZxcwCYxI6sr_4vDKQ/view?usp=sharing). The detection file contain detection result from detector, so that you can run without detector. Create directory ../data/train/lab, put videos and pickle files in it.
run

            python3 master.py
           
