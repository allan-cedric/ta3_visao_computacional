#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import glob as glob

objs = glob.glob('objs/*.jpeg')
scenes = glob.glob('cenas/*.jpeg')

# print(objs)

for img in objs:
    # print(img)
    roi = cv.imread(img)
    assert roi is not None, "file could not be read, check with os.path.exists()"
    hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)

    print(f"Running all scenes to find '{img}'")
    
    for scn in scenes:
        target = cv.imread(scn)
        assert target is not None, "file could not be read, check with os.path.exists()"
        hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)

        # calculating object histogram
        roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

        # normalize histogram and apply backprojection
        cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
        dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

        # Now convolute with circular disc
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        cv.filter2D(dst,-1,disc,dst)

        # threshold and binary AND
        ret,thresh = cv.threshold(dst,50,255,0)
        thresh = cv.merge((thresh,thresh,thresh))
        res = cv.bitwise_and(target,thresh)

        res = np.vstack((target,thresh,res))

        # h, w = res.shape[:2]
        # res_d = cv.resize(res, (w // 4, h // 4), cv.INTER_LINEAR)
        # cv.imshow('output', res_d)
        # cv.waitKey(0)

        output_fname = 'resultados/' + img.split('.')[0].split('/')[1] + '_' + scn.split('/')[1]
        cv.imwrite(output_fname, res)
        # print(output_fname)

        cv.destroyAllWindows()

print("Done! See results folder")
