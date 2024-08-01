# Numeric capture detector
A model of numerical capture reader

# Library used
Pytorch

# Database source
from Kaggle
https://www.kaggle.com/datasets/mahmoudeldebase/captcha-numbers-length-6

# Data Preparation
Total number of data: 1742 png files

Training Set: 0.75*total set = 1306 pic
Valid Set = 87 pic
Testing set = 260 pic

Data preparation in "Grayscale.py"
The program aims to recognize number by number, and 6 digits in one pic

# Modelling

The modelling part is in "Model.py"

The Batch is 30, and 3 layers calculation. 

# Loss

