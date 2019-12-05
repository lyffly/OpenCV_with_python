import cv2
import numpy as np
from matplotlib import pyplot as plt


GOOD_POINTS_LIMITED = 0.986
# 先把原图缩小
is_resized = False

is_sss = True

if is_resized:

    img1 = cv2.imread("a1.jpg",1)
    img2 = cv2.imread("a2.jpg",1)
    print(img1.shape)

    img3 = cv2.resize(img1,(int(4000/5),int(2250/5)), cv2.INTER_CUBIC)
    img4 = cv2.resize(img2,(int(4000/5),int(2250/5)), cv2.INTER_CUBIC)

    cv2.imwrite("a01.jpg",img3)
    cv2.imwrite("a02.jpg",img4)

if is_sss:
    img1 = cv2.imread("a01.jpg",1)
    img2 = cv2.imread("a02.jpg",1)
    #img1 = cv2.GaussianBlur(img1,(3,3),0)
    #img2 = cv2.GaussianBlur(img2,(3,3),0)

    #orb = cv2.ORB_create(200)
    orb = cv2.AKAZE_create()
    kp1,des1 = orb.detectAndCompute(img1, None)
    kp2,des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher_create()
    matches = bf.match(des1, des2)

    matches = sorted(matches,key = lambda x: x.distance)

    goodPoints = []
    for i in range(len(matches) - 1):
        if matches[i].distance < GOOD_POINTS_LIMITED * matches[i+1].distance:
            goodPoints.append(matches[i])

    print(goodPoints)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,goodPoints,flags=2,outImg=None)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1,1,2)

    M,mask = cv2.findHomography(dst_pts,src_pts,cv2.RHO)

    # pic width and height
    h1,w1,p1 = img2.shape
    h2,w2,p2 = img1.shape

    h = np.maximum(h1,h2)
    w = np.maximum(w1,w2)

    _movedis = int(np.maximum(dst_pts[0][0][0], src_pts[0][0][0]))
    imageTransform = cv2.warpPerspective(img2, M, (w1 + w2 - _movedis, h))

    M1 = np.float32([[1,0,0],[0,1,0]])
    h_1,w_1,p = img1.shape

    dst1 = cv2.warpAffine(img1, M1, (w1 + w2 - _movedis, h))

    dst = cv2.add(dst1, imageTransform)
    dst_no = np.copy(dst)

    dst_target = np.maximum(dst1, imageTransform)



    cv2.imshow("image",img3)
    cv2.imshow("dst1",dst1)
    cv2.imshow("dst_target",dst_target)

    cv2.waitKey(0)






















