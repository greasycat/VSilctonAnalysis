import math
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import csv

import pandas as pd

landmarks = {'Batty House': (301.57, -88.5),
             'Lynch Station': (207.35, -204.61),
             'Harris Hall': (69.79, -163.72),
             'Harvey House': (44.48, 118.56),
             'Golledge Hall': (-320.05, -157.11),
             'Snow Church': (-421.37, -156.7),
             'Saucer Center': (-416.92, -295.53),
             'Tobler Museum': (-236.08, -227.27)}

landmarks_order = ["Harris Hall",
                   "Tobler Museum",
                   "Snow Church",
                   "Lynch Station",
                   "Harvey House",
                   "Saucer Center",
                   "Golledge Hall",
                   "Harris Hall",
                   "Batty House"]

number_of_target = len(landmarks_order)


def get_nearest_square_center(pos, unit):
    x, y = pos
    cx = math.copysign(math.ceil(abs(x) / unit) * unit, x)
    cy = math.copysign(math.ceil(abs(y) / unit) * unit, y)
    fx = math.copysign(math.floor(abs(x) / unit) * unit, x)
    fy = math.copysign(math.floor(abs(y) / unit) * unit, y)
    return (cx + fx) / 2, (cy + fy) / 2


def one_to_two(n, row_length):
    return n // row_length, n % row_length


class Wayfinding:

    def __init__(self):
        self.data = {}
        self.shortcuts = {}

    def get_shortcut(self, a, b):
        if a in self.shortcuts and b in self.shortcuts[a]:
            return self.shortcuts[a][b]
        elif b in self.shortcuts and a in self.shortcuts[b]:
            return self.shortcuts[b][a]
        else:
            return None

    def import_everything(self, path):
        files = os.listdir(path)

        # loop through the files and import each file as csv file with delimiter ';'
        for file in files:
            # get file name
            filename = os.fsdecode(file)
            self.data[filename] = np.genfromtxt('data/' + file, delimiter=';')

        with open('shortcuts_fixed.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in reader:
                if self.shortcuts.get(row[0]) is None:
                    self.shortcuts[row[0]] = dict()
                self.shortcuts[row[0]][row[1]] = float(row[2])

    def analyze_all_subject(self, plot=True):
        df = pd.DataFrame(columns=["ParticipantID", "To Tobler Museum", "To Snow Church", "To Lynch Station", "To Harvey House", "To Saucer Center", "To Golledge Hall", "To Harris Hall", "To Batty House"])
        # loop through all subjects
        for subject in self.data:
            self.analyze_subject(subject, plot, df=df)

        # save the data to csv file
        df.to_csv("output.csv", index=False)

    def analyze_subject(self, name, plot=True, df=None):
        # store the first column of the first file in a variable X
        X = self.data[name][:, 0]
        # store the third column of the first file in a variable Z
        Z = self.data[name][:, 2]
        # store the fourth column of the first file in a variable Delta
        Heading = self.data[name][:, 3]

        # combine the X, Z and Delta columns into a matrix with 3 columns
        XZD = np.column_stack((X, Z, Heading))

        # add a column of ones to the back of matrix as the 4th column
        XZD = np.column_stack((XZD, np.ones(len(X))))

        # for each row in the matrix XZD
        # pass the first two columns as the first argument to the function get_nearest_square_center
        # and use 1 as the second argument
        # and replace the first two columns with the returned value
        for i in range(len(XZD)):
            XZD[i, 0:2] = get_nearest_square_center(XZD[i, 0:2], 1)

        # for consecutive rows, calculate the difference between the first and second column of the rows
        # and store the result in a new array
        XZD_diff = np.diff(XZD[:, 0:2], axis=0)
        # print the first 10 rows of the array XZD_diff
        # print(XZD_diff[0:10])

        # if the consecutive rows are the same, delete the second row
        XZD = np.delete(XZD, np.where(np.all(XZD_diff == 0, axis=1)), axis=0)

        # print the first 10 rows of the array XZD
        # print(XZD[0:10])

        indices = []

        # create a list of size number_of_target and fill it with max float
        shortest_distances = [float('inf')] * number_of_target
        decreasing = True
        last_distance = float('inf')
        current_target = 0

        for i in range(len(XZD)):
            if len(indices) == number_of_target:
                break

            x = XZD[i, 0]
            z = XZD[i, 1]
            building = landmarks_order[current_target]
            landmark = landmarks[building]
            distance = sqrt((x - landmark[0]) ** 2 + (z - landmark[1]) ** 2)
            if distance < shortest_distances[current_target]:
                decreasing = True
                shortest_distances[current_target] = distance
            if decreasing and distance > shortest_distances[current_target]:
                decreasing = False
                if last_distance < 20:
                    # print("2. Found", building, "at", i)
                    indices.append(i)
                    current_target += 1

            last_distance = distance

        # Append the last index
        if len(indices) < number_of_target:
            indices.append(len(XZD))

        # print(indices)
        # indices = [5741, 5918]
        # initialize a figure and axes

        sqrt_fig_number = int(sqrt(len(indices) - 1))

        fig, ax = None, None
        if plot:
            fig, ax = plt.subplots(sqrt_fig_number + 1, sqrt_fig_number + 1)

            # load the image "images/silcton_cropped.jpg"
            img = plt.imread('images/silcton_cropped.jpg')

            # rotate the image by 180 degrees
            img = np.rot90(img, 2)

        # ax.imshow(img, extent=[-540, 410, -455, 205])
        # # Use the first column of the XZD matrix as the x axis and the second column as the y axis
        # # and plot the points on the axes
        # ax.plot(XZD[0:2500, 0], XZD[0:2500, 1], 'r-')
        # plt.show()

        # create a dataframe with columns "ParticipantID", "To Tobler Museum", "To Snow Church", "To Lynch Station", "To Harvey House", "To Saucer Center", "To Golledge Hall", "To Harris Hall", "To Batty House"

        distances = [name]
        efficiencies = [name]


        path_distances = []
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]

            # denote the matrix XZD from start to end as XZD_sub with only the first 2 columns
            XZD_sub = XZD[start:end, :2]
            # calculate the sum of absolute differences between consecutive rows in XZD_sub
            # and append the result to path_distances
            path_distances.append(np.sum(np.abs(np.diff(XZD_sub, axis=0))))

            pos = one_to_two(i, sqrt_fig_number + 1)
            if plot:
                ax[pos].imshow(img, extent=[-540, 410, -455, 205])
                ax[pos].plot(XZD[start:end, 0], XZD[start:end, 1], label=i)

            a = landmarks_order[i]
            b = landmarks_order[i + 1]
            distance = path_distances[i]
            shortcut = self.get_shortcut(a, b)
            efficiency = shortcut / distance if distance and shortcut else 0

            if efficiency > 1:
                efficiency = 1

            distances.append(distance)
            efficiencies.append(efficiency)

            print(f"Distance from {a} to {b} is {distance}, shortcut is {shortcut}")
            print(f"Efficiency is {efficiency}")

        # put distances and efficiencies into the dataframe pd as 2 rows
        if df is not None:
            df.loc[len(df)] = distances
            df.loc[len(df)] = efficiencies

        if plot:
            plt.show()

    pass
