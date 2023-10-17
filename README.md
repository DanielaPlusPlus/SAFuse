# SAFuse

Data Augmentation for Deep Visual Recognition Using Superpixel Based Pairwise Image Fusion 
==============================================================================================================

![image](https://github.com/DanielaPlusPlus/SAFuse/blob/main/framework.png)

The source pytorch codes and some trained models are available here.

(1) We discuss the potential shortcomings of existing cutmix-based data augmentation methods from the viewpoint of image fusion.
(2) We present a data augmentation method by fusion of superpixels for the augmented image, and for the first time propose superpixel-attention-based label fusion, which is object-part-aware and efficient.
(3) We propose a novel training framework for a strong classifier with feature fusion and sparse superpixel feature constraints. To the best of our knowledge, it's the first time superpixel-wise contrastive loss with soft labels and weighted local superpixel classification loss are proposed, which is context-aware.
(4) Extensive evaluations on different benchmarks and backbones have been presented and prove the superiority of SAFuse.

Some trained models:
-------------------
SAFuse + DeiT-B/16 + CUB200-2011 + Acc. 82.20%(Link：https://pan.baidu.com/s/1NZ314mXwKnIyzRJMHSxV3Q Extracted code：fkg3)

SAFuse + ResNet50 + Stanford Dogs + Acc. 70.95%(Link：https://pan.baidu.com/s/1vLXVaSefIKtE-RFZG-vceg Extracted code：o5dr)

SAFuse + ResNet50 + CIFAR100 + Acc. 83.92%(Link：https://pan.baidu.com/s/1B5cxhvBcJgiH93Lr5oroRw Extracted code：qaz1)



The top.1 accuracy for classification:
--------------------------------------------------
<table align="left">
  <tr><th align="center">Method</th><th align="center">Dataset</th><th align="center">ResNet18</th><th align="center">ResNeXt50</th></tr>
  <tr><th align="center">OcCaMix</th><th align="center">TinyImageNet</th><th align="center">67.35%</th><th align="center">72.23%</th></tr>
  <tr><th align="center">OcCaMix</th><th align="center">CUB200-2011</th><th align="center">78.40%</th><th align="center">83.69%</th></tr>
  <tr><th align="center">LGCOAMix</th><th align="center">TinyImageNet</th><th align="center">68.27%</th><th align="center">73.08%</th></tr>
  <tr><th align="center">LGCOAMix</th><th align="center">CUB200-2011</th><th align="center">78.87%</th><th align="center">84.37%</th></tr>
</table>
