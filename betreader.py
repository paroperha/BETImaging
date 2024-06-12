'''
Band Edge Thermometry Reader

Paromita Mitchell 07/05/2023
@paroperha

This program takes in data from two cameras to determine transition points of opacity.
This will allow us to determine set points of temperature behavior.
It also serves as a demonstration of temperature dependence of the Band Edge.

There are a few stages to this:
1. Read from two cameras (temperature reading from hotplate, and wafer)
    a) Average the wafers frames.
    b) Save both the temperature and the wafer measurement with filename time.
2. Live feed.
3. Read temperature from picture, or from external temperature sensor.
4. Threshold check for brightness (compare brightness of two points)
5. Edge detect to determine wafer image visibility as a comparison

'''

'''
STAGE 1:
Read from two cameras and keep track of times. This will allow me to make a video
or some such putting this data together. I can manually put the data into a graph.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from imutils.video import VideoStream   # Faster image grabbing threading.





#############################################

srct = 1
srcsc = 2
folder_name = "images/"
file_ext = ".jpg"
test_name = "E1"
save_rate = 1     # seconds before next save
frames_maxlen = 5


# Read from the cameras using VideoStream. 
def stream_init(source):
    return cv2.VideoCapture(index=source)

# Init two cameras
def stream_init_all(srct, srcsc):
    vst = stream_init(srct)
    vssc = stream_init(srcsc)
    return vst, vssc

# Read from a camera
# TODO Check if the frame is good
def read_frame(vs):
    ret, frame = vs.read()
    return frame

def read_frames(vst, vssc):
    return read_frame(vst), read_frame(vssc)

# Output the average of one of the cameras every some amount of time.
def average_frames(frames):
    av = np.average(frames, axis=0)
    av = (np.rint(av)).astype(np.uint8)
    return av

# Display both cameras side by side when function run.
def display_images(temp_frame, sc_frame):
    cv2.imshow('Temp Frame', temp_frame)
    cv2.imshow('Semiconductor Frame', sc_frame)

# Format time
def format_time(t):
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(t))


# Save them into a folder
def save_images(test_name, temp_frame, semi_frame):
    strtime = format_time(time.time())
    prefix = folder_name + test_name + "_" + strtime
    print(folder_name + test_name + "_" + strtime + "_t" + file_ext)
    
    save_image(prefix + "-t" + file_ext, temp_frame)
    save_image(prefix + "-sc" + file_ext, semi_frame)
       
def save_image(name, frame):
    if not cv2.imwrite(name, frame):
        raise Exception("Could not write image: ", name)

# Check for quit
def check_quit(key):

    # if the `q` key was pressed, break from the loop
    return key == ord("q")


# Quit all things safely
def close_all(vst, vssc):
    cv2.destroyAllWindows()

    # TODO Check for all capture and vs
    vst.release()
    vssc.release()



if __name__ == "__main__":

    if len(sys.argv) > 1:
        print("Updating camera sources")
        srct = int(sys.argv[1])
        srcsc = int(sys.argv[2])
    if len(sys.argv) >= 4:
        frames_maxlen = int(sys.argv[3])  # How long to average for
        save_rate = int(sys.argv[4])
    if len(sys.argv) >= 6:
        test_name = str(sys.argv[5])
        exp = float(sys.argv[6])

    vst, vssc = stream_init_all(srct, srcsc)
    
    vssc.set(cv2.CAP_PROP_AUTO_EXPOSURE, exp)

    done = False
    t = ts = ta = time.time()    # t is tracking time, ts is every time image saved.
    tdiff = 0
    frames = []


    frames_add = True
    cont_saving = False

    # Check time, read both frames, display them.
    while not done:
        # Measure time elapsed
        t = time.time()

        # Read and display frames
        temp_frame, sc_frame = read_frames(vst, vssc)

        key = cv2.waitKey(1) & 0xFF

        #TODO Maybe a smarter way to do this
        if len(frames) >= frames_maxlen:
            frames = frames[1:]
        
        if t - ta >= 1/30:
            frames.append(sc_frame)
            ta = t

        # Start and stop continuous saving
        if key == ord("c"):
            if not cont_saving:
                print("Continous saving every", save_rate, "seconds.")
                cont_saving = True
            elif cont_saving:
                print("Stop continuous saving")
                cont_saving = False

        display_images(temp_frame, sc_frame)

        # Don't need to save all frames, can do every few:
        if key == ord("s") or (cont_saving and t-ts >= save_rate):
            print("Saving image at:", format_time(t))

            av_frame = average_frames(frames)
            # display_images(temp_frame, av_frame)

            save_images(test_name, temp_frame, av_frame)
            ts = t

        done = check_quit(key)

    # Close everything


    close_all(vst, vssc)
