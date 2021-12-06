from math import atan, pi, cos
import pandas as pd

# DEFINE CONSTANTS
lat = 40.7781  # Salt Lake City latitude
long = -111.969  # Salt Lake City Longitude
num_quads = 16 # The number of quadrants (0 - 15)
delta = 200  # This is the edge size of each quadrant in km

# CONVERT DIMENSIONS TO DEGREES (LAT/LONG)
delta_deg_lat = delta * 0.009 # 1 degree latitude = 111km => 1/111 = 0.009 degrees per km latitude
delta_deg_long = delta * 0.012 # 1 degree longitude = 84.2km => 1/84.2 = 0.012 degrees per km longitude

# READ IN RAIN AND TEMP CSV
# df = pd.read_csv("rain_and_temp_mini.csv", header='infer')
df = pd.read_csv("rain_and_temp_full.csv", header='infer')
prcp = df.iloc[:, 1]
max_prcp = prcp.max()
tmax = df.iloc[:, 2]
max_overall_temp = tmax.max()

# READ IN WIND CSV
# wind_df = pd.read_csv("wind_mini.csv", header='infer')
wind_df = pd.read_csv("wind_full.csv", header='infer')
wind_direction = wind_df.iloc[:, 4]
wind_speed = wind_df.iloc[:, 3]

# READ IN FIRES CSV
# fires_df = pd.read_csv("fires_mini.csv", header='infer')
fires_df = pd.read_csv("fires_full.csv", header='infer')
fire_size = fires_df.iloc[:, 2]
fire_lat = fires_df.iloc[:, 3]
fire_long = fires_df.iloc[:, 4]
day_start = fires_df.iloc[:, 5]
day_end = fires_df.iloc[:, 6]

# READ IN AIR QUALITY CSV
# df = pd.read_csv("air_mini.csv", header='infer')
df = pd.read_csv("air_full.csv", header='infer')
air_quality_index = df.iloc[:, 3]


# ANGLE ADJUSTER FUNCTION
def angle_adjuster(quadrant_number):
    # Determines if a quadrant is to the top right, bottom right, bottom left or top right from the city
    # Returns the relevant angle adjuster
    if quadrant_number in range(2, 4) or quadrant_number in range(6, 8):
        return pi
    elif quadrant_number in range(0, 2) or quadrant_number in range(4, 6):
        return pi * 2
    elif quadrant_number in range(10, 12) or quadrant_number in range(14, 16):
        return pi
    else:
        return 0


# ASSIGN THE X DIST AND Y DIST IN KM BETWEEN CITY AND EACH QUADRANT CENTRE
# QUADRANTS ARE NUMBERED 0 TO 15 INCLUSIVE
quads_x_km = []
quads_y_km = []
for q in range(num_quads):
    if q % 4 == 0:
        x = -(delta * 1.5)
    elif q % 4 == 1:
        x = -(delta * 0.5)
    elif q % 4 == 2:
        x = delta * 0.5
    else:
        x = delta * 1.5

    if q in range(4):  # ie if it's in the range 0 to 3 inclusive
        y = delta * 1.5
    elif q in range(4, 8):
        y = delta * 0.5
    elif q in range(8, 12):
        y = delta * -0.5
    else:
        y = delta * -1.5

    quads_x_km.append(x)
    quads_y_km.append(y)

quad_to_city_angle_list = []
# CONSIDER 0 DEGREE ANGLE TO BE THE ANGLE GOING FROM ORIGIN ALONG X AXIS IN DIRECTION OF INCREASING X
# CALCULATE THE ANGLE FROM THE CENTRE OF EACH QUADRANT TO THE CITY
# ie IF THE CITY IS DUE NORTHEAST FROM THE CENTRE OF THE QUADRANT THEN THE ANGLE IS PI/4 RADIANS
for q in range(num_quads):
    quad_to_city_angle = (atan(quads_y_km[q] / quads_x_km[q])) + angle_adjuster(q)
    quad_to_city_angle_list.append(quad_to_city_angle)


# Go through each fire and assemble into a day by day feature vector list
num_days = len(wind_direction)
list_of_days = []
for day in range(num_days):
    daily_feature_vector = [0] * num_quads
    list_of_days.append(daily_feature_vector)

for i in range(len(fire_lat)):
    if (long + delta_deg_long) < fire_long[i] <= (long + 2 * delta_deg_long):
        if (lat - 2 * delta_deg_lat) <= fire_lat[i] < (lat - delta_deg_lat):
            quad = 15
        elif (lat - delta_deg_lat) <= fire_lat[i] < lat:
            quad = 11
        elif lat <= fire_lat[i] < (lat + delta_deg_lat):
            quad = 7
        elif (lat + delta_deg_lat) <= fire_lat[i] < (lat + 2 * delta_deg_lat):
            quad = 3

    elif long < fire_long[i] <= (long + delta_deg_long):
        if (lat - 2 * delta_deg_lat) <= fire_lat[i] < (lat - delta_deg_lat):
            quad = 14
        elif (lat - delta_deg_lat) <= fire_lat[i] < lat:
            quad = 10
        elif lat <= fire_lat[i] < (lat + delta_deg_lat):
            quad = 6
        elif (lat + delta_deg_lat) <= fire_lat[i] < (lat + 2 * delta_deg_lat):
            quad = 2

    elif (long - delta_deg_long) < fire_long[i] <= long:
        if (lat - 2 * delta_deg_lat) <= fire_lat[i] < (lat - delta_deg_lat):
            quad = 13
        elif (lat - delta_deg_lat) <= fire_lat[i] < lat:
            quad = 9
        elif lat <= fire_lat[i] < (lat + delta_deg_lat):
            quad = 5
        elif (lat + delta_deg_lat) <= fire_lat[i] < (lat + 2 * delta_deg_lat):
            quad = 1

    elif (long - 2 * delta_deg_long) < fire_long[i] <= (long - delta_deg_long):
        if (lat - 2 * delta_deg_lat) <= fire_lat[i] < (lat - delta_deg_lat):
            quad = 12
        elif (lat - delta_deg_lat) <= fire_lat[i] < lat:
            quad = 8
        elif lat <= fire_lat[i] < (lat + delta_deg_lat):
            quad = 4
        elif (lat + delta_deg_lat) <= fire_lat[i] < (lat + 2 * delta_deg_lat):
            quad = 0

    else:
        quad = 99  # Assign 99 for not in range

    if quad != 99:
        burning_period = range(day_start[i], day_end[i] + 1)

        for day in burning_period:
            if (day < num_days):
                list_of_days[day][quad] += fire_size[i]

max_firewind_feature = 0 # Optionally used later to normalise fire/wind vector values
for day in range(len(wind_direction)):
    raw_wind_angle = wind_direction[day]
    # Wind is reported in degrees and also using a clockwise compass starting at 0 degrees for north
    # Also, wind is reported as the direction it blows from
    # We want to change to the direction it's blowing towards in radians
    corrected_wind_angle = (((raw_wind_angle * -1) + 630) % 360) * ((2 * pi) / 360)
    wind_vector_list = []
    for quad_to_city_angle in quad_to_city_angle_list:
        wind_vector = cos(quad_to_city_angle - corrected_wind_angle) * wind_speed[day]
        wind_vector_list.append(wind_vector)

    # FIRE AREA X WIND VECTOR
    for quad in range(num_quads):
        list_of_days[day][quad] = list_of_days[day][quad] * wind_vector_list[quad]
        if abs(list_of_days[day][quad]) > max_firewind_feature:
            max_firewind_feature = abs(list_of_days[day][quad])

    # APPEND RAIN, TEMPERATURE AND AIR QUALITY
    list_of_days[day].append(prcp[day])
    list_of_days[day].append(tmax[day])
    list_of_days[day].append(air_quality_index[day])

#####################################################################################################################
#####################################################################################################################
# Optional code sections - can comment in or out these as needed:
# Part 1 - convert air quality into either -1 or 1 depending according to threshold
# -1 = good air quality, 1 = not good air quality
AQI_threshold = 50
for day in list_of_days:
    if day[-1] > AQI_threshold:
        day[-1] = 1
    else:
        day[-1] = -1

# Part 2 - normalise wind/fire feature values between -1 and 1
for day in list_of_days:
    for quad in range(num_quads):
        day[quad] = (day[quad] / max_firewind_feature)

# Part 3 - If any wind/fire feature values are negative then change them to 0
for day in list_of_days:
    for quad in range(num_quads):
        if day[quad] < 0:
            day[quad] = 0

# Part 4 - normalise rain and temperature values between 0 and 1
for day in list_of_days:
    day[num_quads] = day[num_quads] / max_prcp
    day[num_quads + 1] = day[num_quads + 1] / max_overall_temp
# Note - the temperature is in the raw data in Kelvin. The normalisation here is scaled from 0 = 0 degrees kelvin
# to 1 = the max recorded temp in kelvin. Can't say for certain this is the best way to do it.

# Part 5 - output all the features to a csv file
with open("feature_file.csv", 'w') as feature_file:
    for day in list_of_days:
        print(*day, sep=',', file=feature_file)