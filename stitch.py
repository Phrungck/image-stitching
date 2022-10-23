import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imageio
import imutils

#Loading the images
query = cv.imread('data/imgpairs/tower_left.jpg')
train = cv.imread('data/imgpairs/tower_right.jpg')

plt_query = imageio.imread('data/imgpairs/tower_left.jpg')
plt_train = imageio.imread('data/imgpairs/tower_right.jpg')

query_img = imageio.imread('data/imgpairs/tower_left.jpg')
train_img = imageio.imread('data/imgpairs/tower_right.jpg')

#initialize variables and arrays
row, col,_ = query_img.shape

#color of corners
red = [255,0,0]

#shows the original images
#fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(10,10))
#ax1.imshow(plt_query)
#ax1.set_xlabel('Query Image')
#ax2.imshow(plt_train)
#ax2.set_xlabel('Train Image')

#plt.show()

#get points in query image
g_query_img = cv.cvtColor(query_img, cv.COLOR_RGB2GRAY)
query_dst = cv.cornerHarris(g_query_img, 7, 9, 0.05)
#query_dst = cv.dilate(query_dst,None)
query_img[query_dst>0.1*query_dst.max()] = red
query_X, query_Y = np.where(np.all(query_img==red,axis=2))
query_pts = np.column_stack((query_X,query_Y))
query_pts = np.float32(query_pts)

#get points in train image
g_train_img = cv.cvtColor(train_img, cv.COLOR_RGB2GRAY)
train_dst = cv.cornerHarris(g_train_img, 7, 9, 0.05)
#train_dst = cv.dilate(train_dst,None)
train_img[train_dst>0.1*train_dst.max()] = red
train_X, train_Y = np.where(np.all(train_img==red,axis=2))
train_pts = np.column_stack((train_X,train_Y))
train_pts = np.float32(train_pts)

#keypoint conversion
kpsTrain = []
kpsQuery = []

def keyConvert(points):
    arr = []

    for i in points:
        x, y = i
        x, y = float(x), float(y)

        kp = cv.KeyPoint(y,x,10)
        arr.append(kp)
    
    return arr

kpsTrain = keyConvert(train_pts)
kpsQuery = keyConvert(query_pts)

#show keypoints
fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(10,10),constrained_layout = False)
ax1.imshow(cv.drawKeypoints(g_train_img, kpsTrain, None,color=(255,0,0)))
ax1.set_xlabel("Keypoints in Train Image")
ax2.imshow(cv.drawKeypoints(g_query_img, kpsQuery, None,color=(255,0,0)))
ax2.set_xlabel("Keypoints in Query Image")

plt.show()

#patch size
size = 8
half = int(size/2)

row0,_ = query_pts.shape  
row1,_ = train_pts.shape

#descriptor arrays
query_ft = []
train_ft = []

def getFeatures(gray_img, points):

    arr = []

    for i in points:
        x, y = i
        x, y = int(x), int(y)

        #Checking if outside image range
        if x-half < 0:
            beginX = 0
            endX = beginX + size
        elif x+half > row:
            endX = row
            beginX = endX - size
        else:
            beginX = x-half
            endX = x+half

        if y-half < 0:
            beginY = 0
            endY = beginY + size
        elif y+half > col:
            endY = col
            beginY = endY - size
        else:
            beginY = y-half
            endY = y+half

        patch = gray_img[beginX:endX,beginY:endY]
        patch = patch.reshape(-1)
        arr = np.append(arr,patch)

    return arr

query_ft = getFeatures(g_query_img, query_pts)
train_ft = getFeatures(g_train_img, train_pts)

#reshaping the vectorized patches
query_ft = query_ft.reshape(row0,-1)
train_ft = train_ft.reshape(row1,-1)

query_ft = np.float32(query_ft)
train_ft = np.float32(train_ft)

#computing the Euclidean distance
bf = cv.BFMatcher(cv.NORM_L2,crossCheck=True)
matches = bf.match(train_ft,query_ft)
#matches = sorted(matches, key = lambda x:x.distance)
#matches = matches[:100]

kpsT = np.float32([kp.pt for kp in kpsTrain])
kpsQ = np.float32([kp.pt for kp in kpsQuery])

ptsA = np.float32([kpsT[m.queryIdx] for m in matches])
ptsB = np.float32([kpsQ[m.trainIdx] for m in matches])

#estimate the affine transformation matrix and rigid mask
a_mat, a_mask = cv.estimateAffinePartial2D(ptsA, ptsB)
a_maskMatches = a_mask.ravel().tolist()
a_draw_params = dict(matchColor = (255,0,0), singlePointColor = None, matchesMask = a_maskMatches, flags = 2)
#Number of inlier points
a_in = a_maskMatches.count(1)
print('Affine inliers: ',a_in)

#getting the average residual of the inliers
def residual(inlier,matches):
    a_sum = np.zeros((1,1))
    ac = 0

    for i in matches:
        if i == 1:
            res = np.square(np.linalg.norm(ptsA[ac]-ptsB[ac]))
            a_sum = a_sum + res
        ac += 1

    ave_res = a_sum/inlier
    return ave_res

a_ave_res = residual(a_in,a_maskMatches)
print('Affine residual: ',a_ave_res)

#draw the matching inliear points
affine_img = cv.drawMatches(plt_train, kpsTrain, plt_query, kpsQuery, matches,None,**a_draw_params)

#estimate the homography between the sets of points
h_mat, h_mask = cv.findHomography(ptsA, ptsB, cv.RANSAC,4)
h_maskMatches = h_mask.ravel().tolist()
h_draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = h_maskMatches, flags =2)
h_in = h_maskMatches.count(1)
print('Homography inliers: ',h_in)

h_ave_res = residual(h_in,h_maskMatches)
print('Homography residual: ', h_ave_res)

#draw the matching inliear points
homog_img = cv.drawMatches(plt_train,kpsTrain,plt_query,kpsQuery,matches,None,**h_draw_params)

fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(10,10),constrained_layout = False)
ax1.imshow(affine_img)
ax1.set_xlabel("Affine Transformation")
ax2.imshow(homog_img)
ax2.set_xlabel("Homography Transformation")
plt.show()

#getting the width and height 
width = train_img.shape[1] + query_img.shape[1]
height = train_img.shape[0] + query_img.shape[0]

#warping the train image based on homography
plt_result = cv.warpPerspective(plt_train, h_mat, (width, height))

#Show warped train image
plt.figure(figsize=(10,10))
plt.imshow(plt_result)

plt.axis('off')
plt.show()

#include query image in plotting
plt_result[0:plt_query.shape[0], 0:plt_query.shape[1]] = plt_query

plt.figure(figsize=(10,10))
plt.imshow(plt_result)

plt.axis('off')
plt.show()

result = cv.warpPerspective(train, h_mat, (width, height))
result[0:query.shape[0], 0:query.shape[1]] = query

#thresholding
gray = cv.cvtColor(result, cv.COLOR_RGB2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

# Finds contours from the binary image
cont = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)

# get the maximum contour area
max_cont = max(cont, key=cv.contourArea)

# get a bbox from the contour area
(x, y, w, h) = cv.boundingRect(max_cont)

# crop the image to the bbox coordinates
result = result[y:y + h, x:x + w]

#this is for saving
#cv.imwrite('stitched_output.jpg', result)