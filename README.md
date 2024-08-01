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

The loss of test dataset is show below. 

![image](https://github.com/user-attachments/assets/8f7abea6-215f-4e52-b1c6-539c8a81587d)


# Accurancy

The best accurate rate is 64.35%, and the predict example by using test dataset is - 

![image](https://github.com/user-attachments/assets/1df9c887-49d4-4550-b570-0174d2c83b93)


4/6 of numbers are predicted successfully. 


