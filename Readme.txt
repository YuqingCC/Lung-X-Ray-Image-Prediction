Predicting Covid Disease Severity and Potential lung Infection Based on X-Ray Image
Author: Yuqing Chen
Notes: This system was completed using Python 3 (ver. 3.8). Please consult Python's version documentations for compability details.

---Required package for this project
   Python 3.8
   matplotlib3.6
   torch1.13
   torch-geometic2.1
   OpenCV(cv2)
   sklearn
   numpy
   tqdm

---Important notes:
The dataset is too large to be included, please refer to https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

---To run
python3 {filename}
For simplicity, I also include pre-trained model in ./model dict. Which can be load by the following:
    state = torch.load(f'./model/model{i}.pth')
main.py: all model performance feature except the class activation map
heatMap.py: the class activation map visualization
** since the class activation map use stored model for label prediction, need to have pre-trained model stored at ./model directory.

--Reference:
https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
https://github.com/zhoubolei/CAM



