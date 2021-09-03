import numpy
import random
from math import cos, sin

def Augmentation(X, Y, angles, noises):
    N = len(X)
    for angle in angles:
        theta = numpy.deg2rad(angle) # set rotation angle
        rot = numpy.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]) # create rotation matrix
        for i in range(0, N):
            data = X[i]; label = Y[i]
            for v in range(0, len(data), 3): # rotate skeleton
                point = numpy.array([data[v], data[v+1]])
                point = numpy.dot(rot, point)
                data[v] = point[0]; data[v + 1] = point[1]
            X.append(data), Y.append(label)
    for noise in noises:
        for i in range(0, N):
            data = X[i]; label = Y[i]
            for v in random.sample(range(0, len(data), 3), 7):  # get random vertexes
                factor = random.uniform(1 - noise, 1 + noise)  # get random noise level within given tolerance
                data[v] = data[v] * factor
                data[v + 1] = data[v + 1] * factor
            X.append(data), Y.append(label)
    return