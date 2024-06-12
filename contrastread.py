'''
Reads a video feed. User chooses points.

Saves all contrast points into a video.

Does a square average around each point.
'''

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from numpy.polynomial import Polynomial as poly

loc = []

# Saves the location of a mouse click.
def save_loc(event, x, y, flags, param):
    global loc
    if event == cv2.EVENT_LBUTTONDOWN:
        loc.append((x,y))
        print(loc)

# Finds the grayscale value of a point on an image.
def find_value(frame, pt):
    x = pt[0]
    y = pt[1]
    return float(frame[y, x])

# Averages a square around the point to use as the diff reference.
def average_square(frame, pt):
    size = 5
    total = 0
    for i in range(pt[0]-size, pt[0]+size):
        for j in range(pt[1]-size, pt[1]+size):
            total += find_value(frame, (i, j))
    av = total/((size*2)**2)
    return av

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):
    global ax

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.subplots_adjust(bottom=0.30)
    plt.title('Contrasts')
    plt.ylabel('Contrast')

def temppoly(temps):
    if len(temps[0])>=2:
        if len(temps[0]) < 7:
            deg = 1
        if len(temps[0]) >= 7:
            deg = 5
        return poly.fit(temps[0], temps[1], deg)
    print("Not enough temp points.")
    return None



def find_contrast(vidfilename, temps=[[], []], live=False):
    global loc
    # Play the video, and track the mouse location.
    cap = cv2.VideoCapture(vidfilename)
    cv2.namedWindow('video')
    cv2.setMouseCallback('video',save_loc)

    # Create an updating graph.
    plt.ion()
    fig, (ax, ax2) = plt.subplots(2, sharex=True)
    hist_plot, = ax.plot([], [])
    xs = []
    contrasts = []

    loc = []

    # Initiate temp graph
    if len(temps[0])>=2:
        if len(temps[0]) < 7:
            deg = 1
        if len(temps[0]) >= 7:
            deg = 5
        ax2.scatter(temps[0], temps[1], label="Temperature data")
        txs, tempfit = poly.fit(temps[0], temps[1]).linspace()
        ax2.plot(txs, tempfit)

    
    # Video Loop
    loop = True
    while loop:
        while(cap.isOpened()):
            ret, img = cap.read()

            # Make sure a there is still video to read.
            if not ret:
                break

            # Convert the vid to grayscale.
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            cv2.imshow('video', img)

            # Prepare a temp graph
            if len(temps[0])>=2:
                ax2.clear()
                ax2.scatter(temps[0], temps[1], label="Temperature data")
                ax2.plot(txs, tempfit)


            # If we have at least 2 points being watched...
            if len(loc) >= 2 and len(loc)%2 == 0:
                # Find the avg val of the latest two points
                v1 = average_square(img, loc[-1])
                v2 = average_square(img, loc[-2])
                diff = abs(v2 - v1)
                contrasts.append(diff)
                
                # Gives a timestamp
                xs.append(cap.get(cv2.CAP_PROP_POS_MSEC))

                ax.clear()
                ax.plot(xs, contrasts, color='blue')
                ax.set_title('Contrasts')
                ax.set_ylabel('Contrast')

            plt.draw()
            plt.pause(0.001)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                loop = False
                break
            # Add temp datapoint
            if pressedKey == ord('t'):
                print("Paused. Grabbing a temperature...")
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                temps[0].append(timestamp)
                temps[1].append(float(input("At "+ str(timestamp) + ", temperature (C): ")))
                # Recalculate polyfit
                if len(temps[0])>=2:
                    if len(temps[0]) < 7:
                        deg = 1
                    if len(temps[0]) >= 7:
                        deg = 5
                    txs, tempfit = poly.fit(temps[0], temps[1], deg).linspace()
            
        # Close and restart the video, best way to loop.
        cap.release()
        cap = cv2.VideoCapture(vidfilename)

    cap.release()
    plt.close()
    cv2.destroyAllWindows()

    loc = []

    return xs, contrasts, temps

def save_temp_contrasts(filename, xs, contrasts, tempfit=False):
    # Create the output lines
    contrast_out = []
    for i in range(len(contrasts)):
        if tempfit:
            temp = tempfit(xs[i])
        else:
            temp = -300 # If no temp fit to work with then throw out rubbish to be manually fixed.
        contrast_out.append(str(temp) + "," + str(contrasts[i]) + "\n")

    # Create the file if it doesn't exist with a header
    if not os.path.isfile(filename):
        with open(filename, "w") as f:
            f.write("Temp (deg C),contrast\n")
    
    # Add the lines
    with open(filename, "a") as f:
        f.writelines(contrast_out)

    print("Saved temperatures and contrasts to", filename)

# Temporary file to interpret later.
def save_temp(filename, temps):
    # Create the output lines
    contrast_out = []
    for i in range(len(temps[0])):
        contrast_out.append(str(temps[0][i]) + "," + str(temps[1][i]) + "\n")

    # Create the file if it doesn't exist with a header
    if not os.path.isfile(filename):
        with open(filename, "w") as f:
            f.write("Timestamp (ms), Temp (deg C)\n")
    
    # Add the lines
    with open(filename, "a") as f:
        f.writelines(contrast_out)

    print("Saved temperatures to", filename)

# Temporary file to interpret later.
def save_contrasts(filename, xs, contrasts):
    # Create the output lines
    contrast_out = []
    for i in range(len(xs)):
        contrast_out.append(str(xs[i]) + "," + str(contrasts[i]) + "\n")

    # Create the file if it doesn't exist with a header
    if not os.path.isfile(filename):
        with open(filename, "w") as f:
            f.write("Timestamp (ms), Contrasts\n")
    
    # Add the lines
    with open(filename, "a") as f:
        f.writelines(contrast_out)

    print("Saved contrasts to", filename)



xs, contrasts, temps = find_contrast("linesvid.mp4")
tempfit = temppoly(temps)
save_temp_contrasts("contrasts.csv", xs, contrasts, tempfit)

xs, contrasts, temps = find_contrast(0)
tempfit = temppoly(temps)
save_temp_contrasts("conlive.csv", xs, contrasts, tempfit)


#TODO Don't save looped values (Maybe don't add vals until loop has restarted)