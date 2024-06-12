# Import a frame.
# Select two points.
# Determine a temperature from dataanalysis.

import cv2
import imageanalysis as iman
import dataanalysis as daan
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    plt.title('Temp from model')
    plt.ylabel('Temperature (deg C)')
    

# Create figure for plotting


# go from existent image, eg frame.png
def find_temp_image(imgfilename, pol):
    global loc
    
    # Setup, find two points
    img = cv2.imread(imgfilename,0)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',save_loc)

    while(1):
        cv2.imshow('image', img)
        # Leave the loop if we hit the 'q' key.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # If we have more than 2 points to compare with, find their diff and the temp.
        if len(loc) >= 2:
            v1 = average_square(img, loc[-1])
            v2 = average_square(img, loc[-2])
            print(v1, v2)
            diff = abs(v2 - v1)
            temp = pol(diff)
            print(diff, temp)

        
    cv2.destroyAllWindows()
    loc = []


# Find temp from a video loop like 'linesvid.mp4' and a polynomial fit.
def find_temp_video_loop(vidfilename, pol):
    global loc
    # Play the video, and track the mouse location.
    cap = cv2.VideoCapture(vidfilename)
    cv2.namedWindow('video')
    cv2.setMouseCallback('video',save_loc)

    # Create an updating graph.
    plt.ion()
    fig, ax = plt.subplots()
    hist_plot, = ax.plot([], [])
    xs = []
    temps = []

    loc = []

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

            # If we have at least 2 points being watched...
            if len(loc) >= 2:
                # Find the avg val of the latest two points
                v1 = average_square(img, loc[-1])
                v2 = average_square(img, loc[-2])
                diff = abs(v2 - v1)
                print(v1, v2, diff)
                
                # Calculate a temp
                temp = pol(diff)
                # Count the graph, just gives an x axis.
                if len(xs) == 0:
                    xs.append(0)
                else:
                    xs.append(xs[-1]+1)
                temps.append(temp)
                print(temp)

                ax.clear()
                ax.plot(xs, temps, color='blue')
                ax.set_title('Modelled temperature')
                ax.set_ylabel('Temperature (C)')
                plt.draw()
                plt.pause(0.001)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                loop = False
                break
            
        # Close and restart the video, best way to loop.
        cap.release()
        cap = cv2.VideoCapture(vidfilename)

    cap.release()
    plt.close()
    cv2.destroyAllWindows()

    loc = []

las = daan.import_fit('904nm_old.csv')
find_temp_image('frame.jpg', las)

led = daan.import_fit("940nm_old.csv")
find_temp_video_loop("linesvid.mp4", las)

find_temp_video_loop(0, led)
