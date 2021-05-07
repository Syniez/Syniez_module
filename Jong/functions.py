import numpy as np
import cv2


def Sobel(img, thres):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    im_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    im_x = cv2.convertScaleAbs(im_x)
    im_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    im_y = cv2.convertScaleAbs(im_y)
    img = cv2.addWeighted(im_x, 1, im_y, 1, 0)

    ret, thresh = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
    return thresh


def Hough(im1, im2, thres):
    # im2 should be grayscale image
    lines = cv2.HoughLinesP(im2, 1, np.pi/180,thres)
    line_img = np.zeros_like(im1)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img,(x1,y1),(x2,y2),[255,0,255],2)

    # blending factor can be adjust
    res = cv2.addWeighted(im1, 0.4, line_img, 0.6, 0)
    return res

    
def SBM(imL, imR, numdisp, blocksize):
    # Both input images are grayscale
    sbm = cv2.StereoBM_create(numDisparities = numdisp, blocksize = blocksize)
    disparity = sbm.compute(imL, imR)
    return disparity


def feature_flow(im1, im2, feature_num):
    feature_params = dict(maxCorners = feature_num, qualityLevel = 0.01, minDistance = 7, blockSize = 7 ) 
    lk_params = dict( winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.zeros([feature_num,3])
    color[:] = [200, 0, 200]

    im1_g = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_g = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1_g = cv2.bilateralFilter(im1_g, 3, 50, 50)
    im2_g = cv2.bilateralFilter(im2_g, 3, 50, 50)

    p0 = cv2.goodFeaturesToTrack(im1_g, mask=None, **feature_params)
    mask = np.zeros_like(im1)
    p1, st, err = cv2.calcOpticalFlowPyrLK(im1_g, im2_g, p0, None,  **lk_params)

    good_new = p1[st == 1] 
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)): 
        a, b = new.ravel() 
        c, d = old.ravel() 
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)       
        right = cv2.circle(im2, (a, b), 5,  color[i].tolist(), -1) 
          
    img = cv2.add(im2, mask)
    return img


def grid_flow(im_g, im, flow, black = False):
    # im : grayscale image
    step = 20
    h, w = im_g.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines + 0.5)
    '''vis = cv2.cvtColor(im_g, cv2.COLOR_GRAY2BGR)
    if black:
        vis = np.zeros((h, w, 3), np.uint8)'''
    cv2.polylines(im, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(im, (x1, y1), 1, (200, 0, 200), -1)
    return im


def Min():
    print("Slave 511... so sad")
