# SAFuse

Data Augmentation for Deep Visual Recognition Using Superpixel Based Pairwise Image Fusion 
==============================================================================================================

![image](https://github.com/DanielaPlusPlus/SAFuse/blob/main/framework.png)

The source pytorch codes and some trained models are available here.

(1) We discuss the potential shortcomings of existing cutmix-based data augmentation methods from the viewpoint of image fusion.
(2) We present a data augmentation method by fusion of superpixels for the augmented image, and for the first time propose superpixel-attention-based label fusion, which is object-part-aware and efficient.
(3) We propose a novel training framework for a strong classifier with feature fusion and sparse superpixel feature constraints. To the best of our knowledge, it's the first time superpixel-wise contrastive loss with soft labels and weighted local superpixel classification loss are proposed, which is context-aware.
(4) Extensive evaluations on different benchmarks and backbones have been presented and prove the superiority of SAFuse.


