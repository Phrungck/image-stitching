# Simple exercise of stitching image pairs (2020 coding-style)

In this programming task, the objective is to use the OpenCV libraries to stitch two images together to form a panoramic image. We aim to implement image stitching
from scratch which uses Harris corner detector as the feature detector.Additionally, RANSAC algorithm is also used to retrieve a homography mapping of one image into the other.

## Dependencies
  * numpy
  * opencv
  * matplotlib
  * imageio
  * imutils

## Procedure
1. Draw keypoints in both images
![alt text](https://github.com/Phrungck/image-stitching/blob/main/keypoints_gray.PNG)
2. Match keypoints using affine or homography with RANSAC
![alt text](https://github.com/Phrungck/image-stitching/blob/main/transformation.PNG)
3. Warp image perspective
![alt text](https://github.com/Phrungck/image-stitching/blob/main/stitched_output.jpg)
