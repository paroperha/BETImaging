'''
imageanalysis.py

This program is a library of functions for image analysis using openCV.

1. Video creation
2. Picture combining
3. Filtering adjustive
4. Averaging of local cells around a given point
5. Reading 7 segment displays

'''

import cv2
import os
import re
import numpy as np
import time

resize_factor = 0.4
roisize = 400
av_radius = 5
bright_x = 220
dark_x = 200

# TODO STAGE 1
# Imports all images with the name and suffix given into an array


def import_frame(name):
    img = cv2.imread("images/"+name, 1)
    return img


def import_frames(name, suf):
    a = []
    for filename in os.listdir("images"):
        name_check = name+"*"+"-"+suf+".jpg"
        if re.match(name+"_[\d-]*"+"-"+suf+".jpg", filename):
            img = cv2.imread("images/"+filename, 1)
            a.append(img)
    return np.array(a)


# Imports all images from one experiment
def import_experiment(name):
    a_t = import_frames(name, "t")
    a_sc = import_frames(name, "sc")

    return a_t, a_sc

# Exports the frames into a folder in frames/<NAME>/
def export_frames(name, a):
    n = 1
    os.makedirs("frames/"+name, exist_ok=True)
    for frame in a:
        cv2.imwrite("frames/"+name+"/"+str(n)+".jpg", frame)
        n += 1

# Creates a video out of frames
def export_video(name, a, fps):
    w = a.shape[2]
    h = a.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"RGBA")
    os.makedirs("videos", exist_ok=True)
    out = cv2.VideoWriter('videos/'+name+'.avi', fourcc, fps, (w, h))

    for frame in a:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

# Just show one image
def show_image(name, frame):
    cv2.imshow(name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Puts the temp at the top left corner, making black invisible, one image
def sc_t_merge(tr,scr):
    # Make a copy to play with
    sc = np.copy(scr)
    t = np.copy(tr)

    # Shrink temp image
    t = cv2.resize(t, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

    # Put logo in top left
    rows, cols, channels = t.shape
    roi = sc[0:rows, 0:cols]

    # Now create a mask of temp and create its inverse mask also
    tgray = cv2.cvtColor(t,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(tgray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    sc_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    t_fg = cv2.bitwise_and(tgray, tgray, mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(sc_bg, t_fg)
    sc[0:rows, 0:cols ] = dst

    return sc

# Does merge for the whole video, probably not efficient.
def sc_t_merge_all(a_t, a_sc):
    a_out = [sc_t_merge(np.array(a_t[i]), np.array(a_sc[i])) for i in range(len(a_t))]

    a_out = np.array(a_out)
    return a_out

# STAGE 2
# Now we read the approximate contrast.

# Find a useful range
def perspectiveWarp(frame, corners):
    rows, cols, ch = frame.shape
    pts1 = np.float32(corners)
    pts2 = np.float32([[0,0], [roisize, 0], [0, roisize], [roisize, roisize]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(frame, M, (roisize, roisize))
    return dst

# Return a balanced image using adaptive threshholding.
def adaptive_thresh(frame):
    th = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY ,11, 10)
    return th

# Read average amplitude around a point
def avg_point(frame, pt):
    x, y = pt
    M = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*av_radius,2*av_radius))

def blur(frame):
    return cv2.bilateralFilter(frame, 9, 100, 100)

# Calculate contrast (subtractive)
def contrast(sqrframe):
    # Choose two points
    dark = sqrframe[80, 320]
    bright = sqrframe[80 + 20, 320 + 20]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sqrframe,'Difference: ' + str(bright - dark),(100, 350), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(sqrframe,'Ratio: ' + str(bright/dark),(100,380), font, 0.5,(255,255,255),1,cv2.LINE_AA)

    return sqrframe, dark, bright

def draw_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param, (x,y), 1, (255, 0, 0), -1)
        print(x, y)
        corners.append([x,y])

# Gets a single point
def get_point(win, frame):
    cv2.setMouseCallback(win, draw_point, param=frame)

# Get bright/dark points from user, marks it.
def get_points(sqrframe):
    circles = cv2.HoughCircles(sqrframe,cv2.HOUGH_GRADIENT,1,20,
    param1=50,param2=30,minRadius=0,maxRadius=0)

    bright = circles + 10

    return circles, bright

def plot_points(sqrframe, pts, color):
    pts = np.uint16(np.around(pts))
    for pt in pts[0,:]:
        cv2.circle(sqrframe, (pt[0], pt[1]), 2, color, -1)

def image_correct(frame, corners):
    warp = perspectiveWarp(frame, corners)
    bw = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    blurred = blur(bw)

    # Set circles to black for image rendering purposes
    val = int(blurred[dark_x, dark_x])
    blurred = blurred - np.minimum(val, blurred)
    return blurred

corners = []

if __name__ == "__main__":
    print("Importing all from gaas940")
    a_t, a_sc = import_experiment("gaas940")
    print(a_t.shape)
    print(a_sc.shape)

    # # Exporting frames to folder
    # print("Exporting frames to folder")
    # export_frames("gaas940ledtest", a_t)

    # # Exporting frames to video
    # print("Exporting video to folder")
    # export_video("gaas940ledtest", a_sc, 20.0)

    # Merge two pictures together
    # print("merging some pictures")
    # res = sc_t_merge(a_t[0], a_sc[0])
    # cv2.imshow("Merged pics",res)
    # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # # Merge all video together
    # print("Merging the videos")
    # a_res = sc_t_merge_all(a_t, a_sc)
    
    # export_video("gaas940ledmerge", a_res, 20.0)

    # # REPEAT FOR 904nm
    # print("Importing all from gaas904")
    # a_t, a_sc = import_experiment("gaas904")
    # print(a_t.shape)
    # print(a_sc.shape)

    # # Merge all video together
    # print("Merging the videos")
    # a_res = sc_t_merge_all(a_t, a_sc)
    
    # export_video("gaas904merge", a_res, 20.0)

    # Find points
    cv2.namedWindow('setup')
    get_point('setup', a_sc[0])
    while(1):
        cv2.imshow('setup', a_sc[0])
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

#     corners = [[358, 247],
# [728, 135],
# [500, 601],
# [911, 446]]

    # Perspective warp and blurred

    a_sc = np.array([contrast(image_correct(frame, corners))[0] for frame in a_sc])

    # Merge all video together
    print("Merging the videos")
    a_res = sc_t_merge_all(a_t, a_sc)
    export_video("gaas940", a_res, 20.0)


    print("Importing all from gaas940")
    a_t, a_sc = import_experiment("gaas940")

    # Find points
    # corners = []
    # cv2.namedWindow('setup')
    # get_point('setup', a_sc[0])
    # while(1):
    #     cv2.imshow('setup', a_sc[0])
    #     if cv2.waitKey(20) & 0xFF == ord("q"):
    #         break
    # cv2.destroyAllWindows()

#     corners = [[123, 249],
# [490, 160],
# [208, 594],
# [630, 472]]

#     # Perspective warp and blurred

#     a_sc = np.array([contrast(image_correct(frame, corners))[0] for frame in a_sc])

#     # Merge all video together
#     print("Merging the videos")
#     a_res = sc_t_merge_all(a_t, a_sc)
#     export_video("gaas940", a_res, 20.0)



























    # # Find circles and points using Haugh
    # dark, bright = get_points(blurred)
    # blurcp = np.copy(blurred)
    # plot_points(blurcp, dark, (0, 255, 0))
    # plot_points(blurcp, bright, (0, 0, 255))
    # show_image("Circle points", blurcp)

    # # Otsu thresholding
    # ret2,th2 = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # show_image("Otsu", th2)
    

    # # Adaptive test
    # print("testing adaptive")
    # res = adaptive_thresh(blurred)
    # cv2.imshow("adaptive pics",res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
