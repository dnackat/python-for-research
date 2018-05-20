#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 00:37:51 2018

@author: dileepn
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

birddata = pd.read_csv("bird_tracking.csv")

bird_names = pd.unique(birddata.bird_name)

# Bird trajectories plot
plt.figure(figsize=(7,7))
for bird_name in bird_names:
    ix = birddata.bird_name == bird_name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    plt.plot(x,y,".",label = bird_name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc = "lower right")
plt.savefig("3traj.pdf")

# Bird speed plot
plt.figure(figsize=(7,7))   
speed = birddata.speed_2d[birddata.bird_name == "Eric"]
ind = np.isnan(speed)
plt.hist(speed[~ind], bins=np.linspace(0,30,20), normed=True)
plt.xlabel("2D speed (m/s)")
plt.ylabel("Frequency")
plt.savefig("hist.pdf")

# Plot using Pandas
birddata.speed_2d.plot(kind='hist', range=[0, 30])
plt.xlabel("2D speed")
plt.savefig("pd_hist.pdf")

# Using Datetime
timestamps = []
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime\
                      (birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S"))

birddata["timestamps"] = pd.Series(timestamps, index = birddata.index)

times = birddata.timestamps[birddata.bird_name == "Eric"]
elapsed_time = [time - times[0] for time in times]

plt.plot(np.array(elapsed_time) / datetime.timedelta(days=1))
plt.xlabel("Observation")
plt.ylabel("Elapsed time (days)")
plt.savefig("timeplotEric.pdf")

# Calculate daily mean speed
data = birddata[birddata.bird_name == "Eric"]
times = data.timestamps
elapsed_time = [time - times[0] for time in times]
elapsed_days = np.array(elapsed_time)/datetime.timedelta(days=1)
next_day = 1
inds = []
daily_mean_speed = []
for (index, time) in enumerate(elapsed_days):
    if time < next_day:
        inds.append(index)
    else:
        # Compute mean speed
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        # Increment next_day by 1
        next_day += 1
        # Reset inds
        inds = []

plt.figure(figsize=(8,6))
plt.plot(daily_mean_speed)
plt.xlabel("Day")
plt.ylabel("Mean speed (m/s)")
plt.savefig("dailyMeanSpeed.pdf")

# Using the Cartopy Library
import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.Mercator()

plt.figure(figsize=(10,10))
# Define plot
ax = plt.axes(projection=proj)
ax.set_extent([-25.0, 20.0, 52.0, 10.0])
# Add more features to plot
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

for name in bird_names:
    ix = birddata["bird_name"] == name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    ax.plot(x, y, '.', transform=ccrs.Geodetic(), label=name)
    
plt.legend(loc="upper left")
plt.savefig("birdmap.pdf")
