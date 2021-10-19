# SRDenseNet in Tensorflow

Tensorflow implementation of [Image Super-Resolution Using Dense Skip Connections](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)

It was trained on the [Div2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) - Train Data (HR images).

## Requirements
- tensorflow
- numpy
- cv2

## SRDenseNet
- (a) SRDenseNet_H : only the hihg-level feature maps are used as input for reconstructing the HR images.
- (b) SRDenseNet_HL : the low-level and the high-level features are combined as input for reconstucting the HR images.
- (c) SRDenseNet_All : all levls of features are combined via skip connections as input for reconstructing the HR images.

![Alt text](images/SRDenseNet.png?raw=true "SRDenseNet architecture")
