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

![image](https://github.com/user-attachments/assets/38b54303-d740-4bf8-a74b-62be22d9945a)

The program aims to recognize number by number, and 6 digits in one pic
so 6 digits with 10 numbers (0, 1, 2, 3 ....) is 6*10, 
and divide to 6 digits
which is 6*10/6

