import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import tau
from scipy.integrate import quad_vec # numerical integration function
from tqdm import tqdm # progress bar library for tracking loops
import matplotlib.animation as animation

# Function to generate x+iy at given time t
def f(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j * y_list)

# Reading the image and converting to grayscale mode
img = cv2.imread("mickey.jpg")
img = cv2.resize(img, (500,500))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
# Applying Canny edge detection
canny_edges = cv2.Canny(blurred, 30, 150)  

# Display the result
cv2.imshow('Canny Edge Detection', canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("mickey_canny_edge.jpg", canny_edges)

# Finding the contours in the Canny edges
contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Merge all contours into one
merged_contour = np.concatenate(contours)

# Displaying the merged contour on the original image
cv2.drawContours(img, [merged_contour], -1, (0, 255, 0), 3)
cv2.imshow('Contours', img)
cv2.imwrite("mickey_merged_contour.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Splitting the coordinate points of the merged contour
x_list = merged_contour[:, 0, 0].astype(np.float64)
y_list = -merged_contour[:, 0, 1].astype(np.float64)

# Centering the contour to origin
x_list -= np.mean(x_list)
y_list -= np.mean(y_list)

# Visualizing the contour
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_list, y_list)
plt.show()

# Time data from 0 to 2*PI as x,y is the function of time
t_list = np.linspace(0, tau, len(x_list))

# Now finding Fourier coefficients
order = 100  # -order to order i.e -100 to 100

print("Generating coefficients ...")
# Computing Fourier coefficients from -order to order
c = []
pbar = tqdm(total=(order * 2 + 1))
# Calculating the coefficients from -order to order
for n in range(-order, order + 1):
    # Calculating definite integration from 0 to 2*PI using formula
    coef = 1 / tau * quad_vec(lambda t: f(t, t_list, x_list, y_list) * np.exp(-n * t * 1j), 0, tau, limit=100,
                               full_output=1)[0]
    c.append(coef)
    pbar.update(1)
pbar.close()
print("Completed generating coefficients.")

# Converting list into numpy array
c = np.array(c)

# Saving the coefficients for later use
np.save("coeff.npy", c)

# Now to make animation with epicycle

# This is to store the points of last circle of epicycle which draws the required figure
draw_x, draw_y = [], []

# Making figure for animation
fig, ax = plt.subplots()

# Different plots to make epicycle
# There are -order to order numbers of circles
circles = [ax.plot([], [], 'r-')[0] for i in range(-order, order + 1)]

# Drawing is plot of final drawing
drawing, = ax.plot([], [], 'k-', linewidth=2)

# Original drawing
orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)

# To fix the size of figure so that the figure does not get cropped/trimmed
ax.set_xlim(np.min(x_list) - 200, np.max(x_list) + 200)
ax.set_ylim(np.min(y_list) - 200, np.max(y_list) + 200)

# Hiding axes
ax.set_axis_off()

# To have symmetric axes
ax.set_aspect('equal')

# Setting up formatting for the video file
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Meem-83 CSE'), bitrate=1800)

print("Compiling animation ...")
# Setting number of frames
frames = 500
pbar = tqdm(total=frames)

# Saving the coefficients in order 0, 1, -1, 2, -2, ...
# It is necessary to make epicycles
def sort_coeff(coeffs):
    new_coeffs = []
    new_coeffs.append(coeffs[order])
    for i in range(1, order + 1):
        new_coeffs.extend([coeffs[order + i], coeffs[order - i]])
    return np.array(new_coeffs)


# Making frame at time t
# t goes from 0 to 2*PI for complete cycle
def make_frame(i, time, coeffs):
    global pbar
    # Getting t from time
    t = time[i]

    # Exponential term to be multiplied with coefficient
    # This is responsible for making rotation of circle
    exp_term = np.array([np.exp(n * t * 1j) for n in range(-order, order + 1)])

    # Sorting the terms of Fourier expression
    coeffs = sort_coeff(coeffs * exp_term)  # coeffs*exp_term makes the circle rotate.
    # Coeffs itself gives only direction and size of circle

    # Splitting into x and y coefficients
    x_coeffs = np.real(coeffs)
    y_coeffs = np.imag(coeffs)

    # Centering points for first circle
    center_x, center_y = 0, 0

    # Making all circles i.e epicycle
    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        # Calculating radius of current circle
        r = np.linalg.norm([x_coeff, y_coeff])  # Similar to magnitude: sqrt(x^2+y^2)

        # Draw circle with given radius at given center points of circle
        # Circumference points: x = center_x + r * cos(theta), y = center_y + r * sin(theta)
        theta = np.linspace(0, tau, num=50)  # Theta should go from 0 to 2*PI to get all points of circle
        x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
        circles[i].set_data(x, y)

        # Calculating center for next circle
        center_x, center_y = center_x + x_coeff, center_y + y_coeff

    # Centering points now are points from last circle
    # These points are used as drawing points
    draw_x.append(center_x)
    draw_y.append(center_y)

    # Drawing the curve from last point
    drawing.set_data(draw_x, draw_y)

    # Drawing the real curve
    orig_drawing.set_data(x_list, y_list)

    # Updating progress bar
    pbar.update(1)


# Making animation
# Time is array from 0 to tau
time = np.linspace(0, tau, num=frames)
anim = animation.FuncAnimation(fig, make_frame, frames=frames, fargs=(time, c), interval=5)
anim.save('mickey.mp4', writer=writer)
pbar.close()
print("Completed: epicycle.mp4")
