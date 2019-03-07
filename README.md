# ann-storm
Code repository for ANN-based multicolor 3D single-molecule localization microscopy
Tensorflow (https://tensorflow.org) library installation is required. (Tested version: Python2.7.12, Tensorflow 1.4.0, Ubuntu 16.04)

# Training

* training_scripts: NN training python scripts for color classification/axial position estimation of a isolated single molecule images.
	* sr_learn_storm_twocolor.py : color classification training (cross-entropy loss)
	* sr_learn_storm_zstack.py : axial localization training (l2 loss)

* training_data: Single molecule images of AF647/CF568 dyes for color classification/axial position estimation (.mat format). Please refer to the Methods section in the manuscript for the detils of the acquisition process.

* pre-trained model checkpoint files for Fig. 4 are available here: https://drive.google.com/open?id=112irADiG5JeXUGZro3Yt_qbZxDkWVM9N 

# Evaluation

* evaluation_scripts: Python scripts for the inferrence of color/z position of the input single molecule images. .ckpt files in /trained_models are used.
	* nn_storm_testing_twocolor.py 	: color classification using trained models in /trained_models, or newly trained models with different training data. "Unknown" images for inference are loaded from /evaluation_data/fig4_color by default and the result is returned to the same folder (/evaluation_data/fig4_color/nn_results_color.mat)
	* nn_storm_testing_z.py	: evaluation code for axial position estimation using models in /trained_models and the data in /evaluation_data/fig4_z.

* evaluation_data: Evaluation data used for Figure 4 in the manuscript. fig4_color and fig4_z has the same content.

* result_summary.m: combine the color classification result & axial localization result and output 3d coordinates for each types of molecules.


Contact: Taehwan Kim (taehwan@eecs.berkeley.edu)
