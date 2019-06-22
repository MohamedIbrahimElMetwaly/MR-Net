# MR-Net
A reproduction of the MR-Net paper by Stanford Machine Learning group. A deep neural network that diagnoses knee MRI scans for abnormalities.
### Colaborators
Mohamed Elsayed Abdelrehim

Mohamed Ibrahim Elmetwally

## Loading the Dataset
For loading the dataset we created a class that extends the dataset class in pytorch and we overrode its __get__(), __len__(), and __init()__ methods such it does not load the whole dataset in memory but rather load only the indexed data at request time. This way we were able to use the whole dataset in training the models. While loading the dataset we took care of any preprocessing required such as normalization and resizing to 3x224x224.
## Creating the Model
The input to MRNet model has dimensions s x 3 x 224 x 224, where s is the number of images in the MRI series (3 is the number of color channels). First, each 3-dimensional MRI image slice was passed through a feature extractor based on AlexNet to obtain a s x 3 x 256 x 6 x 6 tensor containing features for each slice. An adaptive average pooling layer with output 1 was then applied to reduce these features to s x 256. We then applied max pooling across the s dimension to obtain a 256-dimensional vector which was then passed to our fully connected layer which consists of a linear layer with input 256 and output 256 then a dropout layer with probability 0.5 then another layer with input 256 and output 2. The output was then passed to a log softmax function to use with the negative log likelihood criterion function. 
## Training the Models
We trained a total of nine models three detecting abnormalities, three for detecting ACL tears, and three for detecting meniscal tears. Each of each three models were once trained on the coronal exam dataset, the sagittal exam dataset, and the axial exam dataset. We then tried to make an ensemble of each three models to improve the accuracy but it didn’t improve anything.
## Results
The best results we got for detecting abnormalities were 85.8% accuracy and 0.917 f1-score with the axial dataset, for detecting meniscal tears were 70% accuracy and 0.514 f1-score with the coronal dataset, and for detecting ACL tears were 61.7% accuracy and 0.278 f1-score. 

### Original Paper
https://stanfordmlgroup.github.io/projects/mrnet/
