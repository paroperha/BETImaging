
# Finds the temperature from the polynomial models from previous videos.


import numpy as np
from numpy.polynomial import Polynomial as poly
import matplotlib.pyplot as plt

# Use Data analysis to calculate the contrast temperature ratio we expect.
# Use contrast and temp data to calculate polynomial fit.
def poly_temp(contrast, temp, deg):
    pol = poly.fit(contrast, temp, deg=deg)
    return pol

# Import a csv dataset of temp and contrast. Returns two lists of temp and contrasts.
def import_dataset(filename):
    temp = []
    contrast = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            t, c = line.split(",")
            temp.append(float(t))
            contrast.append(float(c))

    return temp, contrast

# Imports and then finds a polynomial based on a dataset csv.
def import_fit(filename, deg=5):
    temp, contrast = import_dataset(filename)
    p = poly_temp(contrast, temp, deg)

    return p


# Returns temp but forces values at edges to be the domain edge.
def find_temp(contrast, pol):
    if contrast<pol.domain[0]:
        print("Warning: value below domain of dataset.")
        return pol(pol.domain[0])
    elif contrast>pol.domain[1]:
        print("Warning: value above domain of dataset.")
        return pol(pol.domain[1])
    return pol(contrast)

def find_lastemp(d):
    if d > 86:
        return 27.8
    lasa = -0.00007932335
    lasb = 0.01155295859
    lasc = -0.6457669288
    lasd = 47.77228515
    temp = lasa*d**3 + lasb*d**2 + lasc*d + lasd
    return temp

def find_ledtemp(d):
    if d > 80:
        return 23.1
    leda = -0.00000012598
    ledb = 0.000023990454
    ledc = -0.00273573329
    ledd = 0.1780350152
    lede = -6.131484009
    ledf = 194.2417133
    temp = leda*d**5 +ledb*d**4 + ledc*d**3 + ledd*d**2 + lede*d + ledf
    return temp

# want to take difference from a frame by choosing two points and then use these functions to give a temperature.

# Test doing so with LED and laser data that I have already.
"""
# Example code using led and laser polynomials.

ledtemp, ledcontrast = import_dataset("904nm_old.csv")
plt.scatter(ledcontrast, ledtemp, color='red', label='Led data')
led = poly_temp(ledcontrast, ledtemp, 5)
print(led)
ledx, ledy = led.linspace()
plt.plot(ledx, ledy, label='Polynomial Fit')
plt.legend()
plt.grid(True)
plt.show()

plt.close()

las = import_fit("940nm_old.csv", deg=3)
print(las)
lasx, lasy = las.linspace()
plt.plot(lasx, lasy, label="Laser fit")
plt.legend()
plt.grid(True)
plt.show()


"""