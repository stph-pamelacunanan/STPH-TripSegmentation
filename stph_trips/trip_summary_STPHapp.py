
import pandas as pd
from geopy.distance import geodesic
from datetime import timedelta
import os
import numpy as np
from io import StringIO
import ast

#######################################################################################################################################

def haversine(lat1, lon1, lat2, lon2):
    """
    DESCRIPTION: To compute the distance, in kilometers, between two points (lon, lat)
    """
    R = 6371  # Radius of Earth in kilometers (change to 3959 for miles)
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    return distance

def compute_speed_with_haversine(df):
    """
    DESCRIPTION: To compute the speed between between two points (lon, lat), given a 'timestamp' column
    """
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Shift column to get previous point
    df['prev_timestamp'] = df['timestamp'].shift(1)

    # Compute time difference in seconds
    df['time_diff_s'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()

    # Compute speed in km/h
    return df['distance_travelled'] / (df['time_diff_s'] / 3600)

def get_overwaiting_durations(df, max_speed = 5, min_duration = 90):
    # Ensure that the timestamp is in the correct datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add a column that indicates whether the vehicle is moving slowly (at most 5 kph)
    df['is_overwaiting'] = df['speed_kph'] <= max_speed
    
    # List to store durations of overwaiting events
    overwaiting_durations = []
    start_time = None
    
    for i in range(1, len(df)):
        # Check for start of a new overwaiting period
        if df['is_overwaiting'].iloc[i] and not df['is_overwaiting'].iloc[i-1]:
            start_time = df['timestamp'].iloc[i]
        
        # If we are in an overwaiting period and the condition is no longer met, check duration
        if not df['is_overwaiting'].iloc[i] and df['is_overwaiting'].iloc[i-1]:
            if start_time is not None:
                duration = (df['timestamp'].iloc[i] - start_time).total_seconds()
                if duration >= min_duration:
                    overwaiting_durations.append(duration)
                start_time = None
    
    # Check if the last segment of data ended with an overwaiting event
    if df['is_overwaiting'].iloc[-1]:
        duration = (df['timestamp'].iloc[-1] - start_time).total_seconds()
        if duration >= min_duration:
            overwaiting_durations.append(duration)
    
    return overwaiting_durations

def get_harsh_acceleration(my_df, acceleration_cutoff):
    df = my_df.copy()

    # Ensure that the timestamp is in the correct datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    # Compute time difference in seconds
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    # Compute speed difference (convert kph to m/s)
    df['speed_mps'] = df['speed_kph'] * (1000 / 3600)  # Convert kph to m/s
    df['speed_diff'] = df['speed_mps'].diff()
    
    # Compute acceleration (m/s^2)
    df['acceleration'] = df['speed_diff'] / df['time_diff']
    
    # Handle potential division by zero or NaN values
    df['acceleration'] = df['acceleration'].fillna(0)
    df['time_diff'] = df['time_diff'].replace(0, np.nan)
    
    # Determine reckless behavior
    reckless = df['acceleration'] > acceleration_cutoff
    
    return reckless.tolist()

def get_harsh_braking(my_df, decceleration_cutoff):
    df = my_df.copy()

    # Ensure that the timestamp is in the correct datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    # Compute time difference in seconds
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    # Compute speed difference (convert kph to m/s)
    df['speed_mps'] = df['speed_kph'] * (1000 / 3600)  # Convert kph to m/s
    df['speed_diff'] = df['speed_mps'].diff()
    
    # Compute acceleration (m/s^2)
    df['acceleration'] = df['speed_diff'] / df['time_diff']
    
    # Handle potential division by zero or NaN values
    df['acceleration'] = df['acceleration'].fillna(0)
    df['time_diff'] = df['time_diff'].replace(0, np.nan)
    
    # Determine reckless behavior
    braking = df['acceleration'] < decceleration_cutoff
    
    return braking.tolist()

def get_overspeeding_duration(df: pd.DataFrame, overspeeding: float, maxOverspeed: float = None) -> float:
    """
    Calculate the number of seconds the vehicle was overspeeding.
    
    Parameters:
    df (pd.DataFrame): DataFrame with 'speedInKph' and 'timestamp' columns.
    overspeeding (float): Overspeeding threshold in kph.
    maxOverspeed (float, optional): Maximum speed limit to consider for overspeeding duration.
    
    Returns:
    float: Total duration in seconds the vehicle was overspeeding.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    if maxOverspeed is not None:
        overspeeding_duration = df.loc[(df['speed_kph'] > overspeeding) & (df['speed_kph'] <= maxOverspeed), 'time_diff'].sum()
    else:
        overspeeding_duration = df.loc[df['speed_kph'] > overspeeding, 'time_diff'].sum()
    
    return overspeeding_duration if not np.isnan(overspeeding_duration) else 0

def compute_missing_data_proportion(df):
    # Ensure that the timestamp is in the correct datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get the earliest and latest timestamps in the dataset
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    
    # Create a sequence of all expected timestamps (1 second intervals)
    expected_timestamps = pd.date_range(start=start_time, end=end_time, freq='1s')
    
    # Get the set of actual timestamps from the dataset
    actual_timestamps = set(df['timestamp'])
    
    # Count the missing timestamps
    missing_timestamps = expected_timestamps.difference(actual_timestamps)
    
    # Calculate the total number of expected timestamps
    total_seconds = len(expected_timestamps)
    
    # Proportion of missing data
    missing_proportion = len(missing_timestamps) / total_seconds
    
    return round(missing_proportion,2)

#######################################################################################################################################

def tripSuperSummary_STPHapp(vehicle_feeds_with_tripID, speed_cutoff = 5,
                             overwaiting_time = [90, 150],
                             overspeeding_thresholds = [60, 65],
                             harsh_acceleration = [2.5, 3.5],
                             harsh_braking = [-2.5, -3.5]):
    """
    vehicle_feeds_with_tripID: pd.DataFrame that is a result of the trip_segmentation() function (i.e., feeds from a single vehicle)
    """
    
    my_trips_summary = pd.DataFrame()

    # Breakdown the df into one df per trip, saved in a list
    list_of_trips = []
    for trip_id in vehicle_feeds_with_tripID['trip_identifier'].unique().tolist():
        df = vehicle_feeds_with_tripID[vehicle_feeds_with_tripID['trip_identifier'] == trip_id]
        ## Ensure df is ordered chronologically
        df = df.sort_values(by = 'timestamp').reset_index(drop=True)
        list_of_trips.append(df)
    
    for df in list_of_trips:
        df = df.drop(columns = ['distance_travelled'])   # This will be recomputed
        
        # Recompute distances travelled per row
        df['prev_latitude'] = df['latitude'].shift(1)
        df['prev_longitude'] = df['longitude'].shift(1)
        df['distance_travelled'] = df.apply(lambda x: haversine(x['prev_latitude'], x['prev_longitude'],
                                                               x['latitude'], x['longitude']), axis=1)
        df['distance_travelled'] = df['distance_travelled'].fillna(0)
        df['speed_kph'] = compute_speed_with_haversine(df)   # Adding a speed column
        
        # Compute the stats
        last_row = len(df)-1
        ## Metadata
        plate_number = df['plate_number'].values[0]
        route = df['route'].values[0]
        trip_type = df['trip_identifier'].str.split("_").str[0].str.capitalize().values[0]
        trip_type2 = df['trip_identifier'].str.split("_").str[1].str.capitalize().values[0]
        if trip_type2 == 'Trip':
            trip_status = 'Complete trip'
        else:
            trip_status = 'Cut trip'
        date = df['timestamp'].dt.strftime("%b %d, %Y")[0]
        start = df['timestamp'].dt.strftime("%I:%M %p")[0]
        end = df['timestamp'].dt.strftime("%I:%M %p")[last_row]
        ## Stats
        duration = (df['timestamp'][last_row] - df['timestamp'][0]).total_seconds() / 60
        ave_speed = df['speed_kph'].mean()
        max_speed = df['speed_kph'].max()
        distance_travelled = df['distance_travelled'].sum()
        missing_data = compute_missing_data_proportion(df)
        
        # Append to the dataframe
        new_row = pd.DataFrame(data = {'Vehicle ID': [plate_number],
                                       'Route': [route],
                                       'Trip type': [trip_type],
                                       'Trip status': [trip_status],
                                       'Date': [date],
                                       'Start time': [start], 'End time': [end],
                                       'Total trip duration (min)': [round(duration,2)],
                                       'Average speed (kph)': [float(round(ave_speed, 2))],
                                       'Maximum speed (kph)': [float(round(max_speed, 2))],
                                       'Total distance travelled (km)': [float(round(distance_travelled, 2))],
                                       'Missing data (%)': [missing_data * 100]})

        # For criteria with several thresholds
        
        ## Overwaiting
        for overwaiting_threshold in overwaiting_time:
            overwaiting = get_overwaiting_durations(df, max_speed = speed_cutoff, min_duration = overwaiting_threshold)
            overwaiting_events = len(overwaiting)
            if overwaiting_events>0:
                ave_overwaiting_duration = float(round(np.mean(overwaiting)))
            else:
                ave_overwaiting_duration = 0
            colname1 = 'Total overwaiting events (' + str(overwaiting_threshold) + ')'
            new_row[colname1] = overwaiting_events
            colname2 = 'Average overwaiting time (' + str(overwaiting_threshold) + ')'
            new_row[colname2] = ave_overwaiting_duration

        ## Harsh acceleration
        for acceleration in harsh_acceleration:
            harsh_acceleration_result = get_harsh_acceleration(df, acceleration)
            colname3 = 'Total harsh acceleration events ' + str(acceleration)
            new_row[colname3] = sum(harsh_acceleration_result)
        
        ## Harsh braking
        for braking in harsh_braking:
            harsh_braking_result = get_harsh_braking(df, braking)
            colname4 = 'Total harsh braking events ' + str(braking)
            new_row[colname4] = sum(harsh_braking_result)

        ## Overspeeding
        if len(overspeeding_thresholds) > 1:
            for i in range(len(overspeeding_thresholds)): 
                overspeed = overspeeding_thresholds[i]
                if i+1 != len(overspeeding_thresholds):
                    next_overspeed = overspeeding_thresholds[i+1]
                    overspeeding_duration = get_overspeeding_duration(df, overspeed, next_overspeed)
                    colname5 = 'Total overspeeding duration (' + str(overspeed) + '-' + str(next_overspeed) + ' kph)'
                    new_row[colname5] = overspeeding_duration
                else:
                    overspeeding_duration = get_overspeeding_duration(df, overspeed)
                    colname5 = 'Total overspeeding duration (' + str(overspeed) + ' kph)'
                    new_row[colname5] = overspeeding_duration
        else:
            for overspeed in overspeeding_thresholds: 
                overspeeding_duration = get_overspeeding_duration(df, overspeed)
                colname5 = 'Total overspeeding duration (' + str(overspeed) + ' kph)'
                new_row[colname5] = overspeeding_duration

        # Add the trip_id for backtracking
        new_row['Trip ID'] = df['trip_identifier'].values[0]

        # Add to the main dataframe result
        my_trips_summary = pd.concat([my_trips_summary, new_row], ignore_index=True)
        
    # Return the trip super summaries table
    return my_trips_summary
