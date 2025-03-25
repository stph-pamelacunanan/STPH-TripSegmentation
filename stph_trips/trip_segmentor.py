
import pandas as pd
from geopy.distance import geodesic
import os
import numpy as np
from io import StringIO
import math
import folium
import ast
from scipy.spatial import cKDTree

def haversine(lat1, lon1, lat2, lon2):
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

def txt_to_df(filename):
    try:
        # Open the file in read mode
        with open(filename, 'r') as file:
            # Read the contents of the file
            contents = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    # Convert to a dataframe
    ## Remove the BOM character if present
    data = contents.lstrip('\ufeff')
    ## Use StringIO to simulate a file-like object
    data_io = StringIO(data)
    ## Read the data into a pandas DataFrame
    df = pd.read_csv(data_io)
    return df

def interpolate_points(start, end, max_distance):
    """Generate intermediate points between two points if they are too far apart."""
    points = [start]  
    start_coords = np.array(start)
    end_coords = np.array(end)

    total_distance = geodesic(start, end).meters
    num_extra_points = int(np.floor(total_distance / max_distance))

    if num_extra_points > 0:
        for i in range(1, num_extra_points + 1):
            ratio = i / (num_extra_points + 1)
            interpolated_point = start_coords + ratio * (end_coords - start_coords)
            points.append(tuple(interpolated_point))

    points.append(end)  
    return points

def reduce_gps_points(df, lat_col, lon_col, min_distance=100, max_distance=150):
    """
    Reduce GPS points such that:
    - Any two retained points are at least `min_distance` meters apart.
    - Intermediate points are inserted if two points are more than `max_distance` meters apart.

    :param df: Pandas DataFrame containing latitude and longitude columns.
    :param lat_col: Name of the latitude column.
    :param lon_col: Name of the longitude column.
    :param min_distance: Minimum distance (in meters) between retained points.
    :param max_distance: Maximum distance (in meters) between retained points.
    :return: Filtered Pandas DataFrame with all original columns.
    """
    if df.empty:
        return df

    points = list(df.itertuples(index=False, name=None))
    lat_idx = df.columns.get_loc(lat_col)
    lon_idx = df.columns.get_loc(lon_col)

    selected = [points[0]]  # Start with the first point
    
    for point in points[1:]:
        lat_lon = (point[lat_idx], point[lon_idx])
        last_selected = (selected[-1][lat_idx], selected[-1][lon_idx])

        distance_to_last = geodesic(lat_lon, last_selected).meters

        if distance_to_last >= max_distance:
            # Insert intermediate points if the gap is too large
            interpolated = interpolate_points(last_selected, lat_lon, max_distance)
            selected.extend([(None, *pt) for pt in interpolated[1:-1]])

        if distance_to_last >= min_distance:
            selected.append(point)

    # Final pass to ensure distance conditions are met
    final_selected = [selected[0]]
    for point in selected[1:]:
        lat_lon = (point[lat_idx], point[lon_idx])
        last_selected = (final_selected[-1][lat_idx], final_selected[-1][lon_idx])

        distance_to_last = geodesic(lat_lon, last_selected).meters

        if distance_to_last >= max_distance:
            interpolated = interpolate_points(last_selected, lat_lon, max_distance)
            final_selected.extend([(None, *pt) for pt in interpolated[1:-1]])

        if distance_to_last >= min_distance:
            final_selected.append(point)

    # Convert selected points back to DataFrame
    selected_df = pd.DataFrame(final_selected, columns=df.columns)

    return selected_df

def obtain_route_dict(path_of_gtfs_shapefiles, min_dist = 100, max_dist = 150):
    txt_files = [file for file in os.listdir(path_of_gtfs_shapefiles) if file.endswith('.txt')]

    # Initialize a dictionary in the form:
    ## {'route_name': {'inbound': inbound_stops_df, 'outbound': outbound_stops_df}}
    routes_dict = {}
    
    for txt_file in txt_files:
        file_path = os.path.join(path_of_gtfs_shapefiles, txt_file)
        
        # Get the df
        my_route = txt_to_df(file_path)
        
        # Outbound trip (shape_id = 1)
        ## Refine the GTFS points of the outbound route
        outbound_route = reduce_gps_points(my_route[my_route['shape_id'] == 1],
                                           'shape_pt_lat', 'shape_pt_lon',
                                           min_distance = min_dist, max_distance = max_dist)
        ## Fix the sequencing of stops
        outbound_route = outbound_route.reset_index().drop(columns = ['shape_pt_sequence']).rename(columns = {'index': 'stop_id'})
        outbound_route['stop_id'] = outbound_route['stop_id'] + 1
        outbound_route = outbound_route.rename(columns = {'shape_pt_lat': 'stop_lat', 'shape_pt_lon': 'stop_lon'})
        ## Add a distance travelled column (in km)
        outbound_route['prev_latitude'] = outbound_route['stop_lat'].shift(1)
        outbound_route['prev_longitude'] = outbound_route['stop_lon'].shift(1)
        outbound_route['kmTravelled'] = outbound_route.apply(lambda x: haversine(x['prev_latitude'], x['prev_longitude'],
                                                                                 x['stop_lat'], x['stop_lon']), axis=1)
        outbound_route['kmTravelled'] = outbound_route['kmTravelled'].fillna(0)
        outbound_route = outbound_route.drop(columns = ['prev_latitude', 'prev_longitude', 'shape_id'])
    
        # Inbound trip (shape_id = 2)
        ## Refine the GTFS points of the inbound route
        inbound_route = reduce_gps_points(my_route[my_route['shape_id'] == 2],
                                          'shape_pt_lat', 'shape_pt_lon',
                                          min_distance = min_dist, max_distance = max_dist)
        ## Fix the sequencing of stops
        inbound_route = inbound_route.reset_index().drop(columns = ['shape_pt_sequence']).rename(columns = {'index': 'stop_id'})
        inbound_route['stop_id'] = inbound_route['stop_id'] + 1
        inbound_route = inbound_route.rename(columns = {'shape_pt_lat': 'stop_lat', 'shape_pt_lon': 'stop_lon'})
        ## Add a distance travelled column (in km)
        inbound_route['prev_latitude'] = inbound_route['stop_lat'].shift(1)
        inbound_route['prev_longitude'] = inbound_route['stop_lon'].shift(1)
        inbound_route['kmTravelled'] = inbound_route.apply(lambda x: haversine(x['prev_latitude'], x['prev_longitude'],
                                                                               x['stop_lat'], x['stop_lon']), axis=1)
        inbound_route['kmTravelled'] = inbound_route['kmTravelled'].fillna(0)
        inbound_route = inbound_route.drop(columns = ['prev_latitude', 'prev_longitude', 'shape_id'])
        
        # Set the route name as the key
        route_name = txt_file.removesuffix(".txt")
        
        # Append it in the dictionary `route_dict`
        routes_dict[route_name] = {'outbound': outbound_route, 'inbound': inbound_route}

    return routes_dict

#######################################################################################################################################

def route_gtfs_stops_mapper(df, latitude_name = 'latitude', longitude_name = 'longitude',
                            output_html = "custom_markers_map.html"):
    """
    Maps the coordinates from a DataFrame, connects them with a black line, 
    and customizes the markers for the first and last points.

    Parameters:
        df (pd.DataFrame): DataFrame with 'latitude' and 'longitude' columns.
        output_html (str): File name for the output HTML.
    """
    if latitude_name not in df.columns or longitude_name not in df.columns:
        raise ValueError("DataFrame must contain the specified latitude and longitude column names.")
    
    # Create a map centered at the first point
    map_obj = folium.Map(location=[df[latitude_name][0], df[longitude_name][0]], zoom_start=18)

    # Draw a black line connecting all the points
    coordinates = list(zip(df[latitude_name], df[longitude_name]))
    folium.PolyLine(
        locations=coordinates,
        color='black',  # Line color
        weight=2,       # Line thickness
    ).add_to(map_obj)

    # Add CircleMarkers with custom colors for the first and last points
    for i, (lat, lon) in enumerate(coordinates):
        if i == 0:
            color = 'green'  # First point
        elif i == len(df) - 1:
            color = 'red'  # Last point
        else:
            color = 'black'  # Other points

        folium.CircleMarker(
            location=(lat, lon),
            radius=2.5,  # Control the size of the circle
            color=color,  # Circle border color
            fill=True,
            fill_color=color,  # Circle fill color
            fill_opacity=1.0,  # Opacity of the fill
        ).add_to(map_obj)

    # Save the map as an HTML file
    map_obj.save(output_html)
    print(f"Map saved as {output_html}")

#######################################################################################################################################

def nearest_stop_checker(vehicle_feeds_df, routes_dict, trip_type, dist_cutoff = 50):
    """
    Finds the nearest stop for each GPS point in vehicle_feeds_df within a 50-meter radius
    and returns the original DataFrame with an added 'stop_id' column.

    :param vehicle_feeds_df: DataFrame with columns ["latitude", "longitude", "route"].
    :param trip_type: string, either 'inbound' or 'outbound'
    :return: Original vehicle_feeds_df with an added 'stop_id' column.
    """
    
    # Obtain the `route_stops_df`
    route_id = vehicle_feeds_df['route'].values[0]
    route_stops_df = routes_dict[route_id][trip_type]
    
    # Convert stop coordinates into a fast lookup structure (KDTree)
    stop_coords = route_stops_df[["stop_lat", "stop_lon"]].to_numpy()
    stop_tree = cKDTree(stop_coords)   ## Efficient spatial search structure
    
    # Convert vehicle coordinates to numpy array
    vehicle_coords = vehicle_feeds_df[["latitude", "longitude"]].to_numpy()
    
    # Query KDTree for the nearest stop for each vehicle point
    distances, indices = stop_tree.query(vehicle_coords, k=1)   ## Get closest stop index
    
    # Compute geodesic distances to verify it's within 50 meters
    closest_stops = []
    for i, (vehicle_point, stop_idx) in enumerate(zip(vehicle_coords, indices)):
        stop_point = stop_coords[stop_idx]
        distance = geodesic(vehicle_point, stop_point).meters   ## Compute distance
        
        if distance <= dist_cutoff:
            closest_stops.append(route_stops_df.iloc[stop_idx]["stop_id"])
        else:
            closest_stops.append(0)   ## No stop within range
    
    # Append stop_id column to the original DataFrame
    vehicle_feeds_df = vehicle_feeds_df.copy()
    vehicle_feeds_df["stop_id"] = closest_stops

    # Assign NA to "stop_id" if the "identifier" column is non-NA
    vehicle_feeds_df.loc[vehicle_feeds_df['identifier'] != '', 'stop_id'] = np.nan
    
    return vehicle_feeds_df[['imei', 'timestamp', 'latitude', 'longitude', 'distanceTravelled',
                             'stop_id', 'route', 'identifier']]

def sequence_checker(df, trip_type, zero_cutoff = 60):
    """
    Groups stop_id sequences that are non-decreasing and assigns a trip identifier.
    Stops counting when encountering NaN in stop_id and does not start new sequences with NaN.

    :param df: DataFrame with columns ["timestamp", "latitude", "longitude", "stop_id"]
    :param trip_type: string, either 'inbound' or 'outbound'
    :param zero_cutoff: Number of consecutive zeros to stop the trip sequence. 60 to represent 60 secs or 1 min.
    :return: DataFrame with an additional "identifier" column.
    """
    df = df.copy()

    current_identifier = None
    start_timestamp = None
    last_stop_id = 0
    zero_count = 0
    in_sequence = False

    for i, row in df.iterrows():
        stop_id = row["stop_id"]
        
        # Stop sequence if stop_id is NaN
        if pd.isna(stop_id):
            in_sequence = False
            current_identifier = None
            zero_count = 0  # Reset zero counter
            continue
        
        # If stop_id is zero, count consecutive zeros
        if stop_id == 0:
            zero_count += 1
            if zero_count > zero_cutoff and in_sequence:
                in_sequence = False  # Stop sequence
                current_identifier = None
            continue
        else:
            zero_count = 0  # Reset zero counter if stop_id is nonzero

        # Start a new sequence if:
        # 1. We are not in a sequence
        # 2. stop_id is nonzero and NOT NaN
        if not in_sequence and (stop_id > 0  and pd.notna(stop_id)):
            start_timestamp = row["timestamp"]
            current_identifier = f"{trip_type}_trip_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
            in_sequence = True

        # If we are in a sequence, ensure non-decreasing stop_id
        if in_sequence:
            if stop_id < last_stop_id:  # If stop_id decreases, stop the sequence
                in_sequence = False
                current_identifier = None
            else:
                df.at[i, "identifier"] = current_identifier  # Assign trip ID

        last_stop_id = stop_id  # Update last stop_id

    return df

def trip_validator(trips, trip_type, routes_dict, dist_threshold = 0.7, time_threshold = 15):
    
    # Get the expected distance from the trip_type and route
    route_id = trips['route'].values[0]
    route_stops_df = routes_dict[route_id][trip_type]
    expected_dist = float(route_stops_df['kmTravelled'].sum())   
    
    # Obtain valid entries
    trip_summary = trips[trips['identifier'] != ''].groupby('identifier').agg(
        start_time = ('timestamp', 'min'),
        time_diff = ('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        distance = ('distanceTravelled', 'sum'))
    trip_summary = trip_summary.reset_index()
    valid_trips = trip_summary[(trip_summary['time_diff'] > time_threshold) & \
                    (trip_summary['distance'] >= (dist_threshold * expected_dist))].reset_index(drop=True)

    # Return the identifiers of the valid trips
    return valid_trips['identifier'].values.tolist()

def cut_trips_determinant(trips, trip_type, routes_dict, dist_threshold = 0.7):
    
    # Get the expected distance from the trip_type and route
    route_id = trips['route'].values[0]
    route_stops_df = routes_dict[route_id][trip_type]
    expected_dist = float(route_stops_df['kmTravelled'].sum())   
    
    # Obtain cut trips
    trip_summary = trips[trips['identifier'] != ''].groupby('identifier').agg(
        start_time = ('timestamp', 'min'),
        time_diff = ('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        distance = ('distanceTravelled', 'sum'))
    trip_summary = trip_summary.reset_index()
    cut_trips = trip_summary[(trip_summary['distance'] >= 1) & \
                    (trip_summary['distance'] < (dist_threshold * expected_dist))].reset_index(drop=True)

    # Return the identifiers of the cut trips
    return cut_trips['identifier'].values.tolist()

def trip_segmentation(vehicle_feeds_df, routes_dict, my_dist_cutoff, zero_cutoff, my_dist_threshold, my_time_threshold):
    
    # Step 1
    vehicle_feeds = vehicle_feeds_df.copy()
    vehicle_feeds_engineOn = vehicle_feeds[vehicle_feeds['engineRpm'] > 0].reset_index(drop=True)
    vehicle_feeds_engineOn['identifier'] = ''   ## Initialize an empty "identifier" column (similar to a trip_id)
    vehicle_feeds_engineOn = vehicle_feeds_engineOn.sort_values(by = ['timestamp'])   ## Ensure data is in chronological order

    ################################## ------------ OUTBOUND TRIPS ------------ ##################################
    
    if len(vehicle_feeds_engineOn) > 0:
        # Step 2
        outbound = nearest_stop_checker(vehicle_feeds_df = vehicle_feeds_engineOn,
                                        routes_dict = routes_dict, 
                                        trip_type = 'outbound', dist_cutoff = my_dist_cutoff)

        # Step 3
        outbound_trips = sequence_checker(outbound, trip_type = 'outbound')

        # Step 4
        outbound_completeTrips_identifiers = trip_validator(outbound_trips, trip_type = 'outbound',
                                                            routes_dict = routes_dict, 
                                                            dist_threshold = my_dist_threshold, time_threshold = my_time_threshold)

        # Step 5
        outbound_cutTrips_identifiers = cut_trips_determinant(outbound_trips, trip_type = 'outbound',
                                                            routes_dict = routes_dict, 
                                                            dist_threshold = my_dist_threshold)

        # Step 6
        outbound_completeTrips = outbound_trips.loc[outbound_trips['identifier'].isin(outbound_completeTrips_identifiers),
                                                    ['imei', 'timestamp', 'longitude', 'latitude',
                                                    'identifier']].reset_index(drop=True)
        outbound_cutTrips = outbound_trips.loc[outbound_trips['identifier'].isin(outbound_cutTrips_identifiers),
                                                ['imei', 'timestamp', 'longitude', 'latitude',
                                                'identifier']].reset_index(drop=True)
        outbound_cutTrips['identifier'] = outbound_cutTrips['identifier'].str.replace("trip", "cuttrip", regex=False)
        
        outbound_trips = pd.concat([outbound_completeTrips, outbound_cutTrips], ignore_index=True)
        vehicle_feeds_engineOn = vehicle_feeds_engineOn.drop(columns = ['identifier'])
        vehicle_feeds_engineOn = vehicle_feeds_engineOn.merge(outbound_trips,
                                                            on = ['imei', 'timestamp', 'longitude', 'latitude'],
                                                            how = 'left')
        vehicle_feeds_engineOn['identifier'] = vehicle_feeds_engineOn['identifier'].fillna('')

        ################################## ------------ INBOUND TRIPS ------------ ##################################

        # Step 2
        inbound = nearest_stop_checker(vehicle_feeds_df = vehicle_feeds_engineOn,
                                    routes_dict = routes_dict, 
                                    trip_type = 'inbound', dist_cutoff = my_dist_cutoff)

        # Step 3
        inbound_trips = sequence_checker(inbound, trip_type = 'inbound')

        # Step 4
        inbound_completeTrips_identifiers = trip_validator(inbound_trips, trip_type = 'inbound',
                                                        routes_dict = routes_dict, 
                                                        dist_threshold = my_dist_threshold, time_threshold = my_time_threshold)

        # Step 5
        inbound_cutTrips_identifiers = cut_trips_determinant(inbound_trips, trip_type = 'inbound',
                                                            routes_dict = routes_dict, 
                                                            dist_threshold = my_dist_threshold)

        # Step 6
        inbound_completeTrips = inbound_trips.loc[inbound_trips['identifier'].isin(inbound_completeTrips_identifiers),
                                                ['imei', 'timestamp', 'longitude', 'latitude',
                                                'identifier']].reset_index(drop=True)
        inbound_cutTrips = inbound_trips.loc[inbound_trips['identifier'].isin(inbound_cutTrips_identifiers),
                                            ['imei', 'timestamp', 'longitude', 'latitude',
                                            'identifier']].reset_index(drop=True)
        inbound_cutTrips['identifier'] = inbound_cutTrips['identifier'].str.replace("trip", "cuttrip", regex=False)
        
        inbound_trips = pd.concat([inbound_completeTrips, inbound_cutTrips], ignore_index=True)

        vehicle_feeds_engineOn = vehicle_feeds_engineOn.merge(inbound_trips,
                                                            on = ['imei', 'timestamp', 'longitude', 'latitude'],
                                                            how = 'left', 
                                                            suffixes = ("_outbound", "_inbound"))
        # Fix the identifier column (it has been doubled)
        vehicle_feeds_engineOn["trip_identifier"] = np.where(vehicle_feeds_engineOn["identifier_outbound"] == "",
                                            vehicle_feeds_engineOn["identifier_inbound"],
                                            vehicle_feeds_engineOn["identifier_outbound"])
        vehicle_feeds_engineOn = vehicle_feeds_engineOn.drop(columns = ['identifier_outbound', 'identifier_inbound'])

        # Return only the vehicle feeds with identified trip
        return vehicle_feeds_engineOn[(vehicle_feeds_engineOn['trip_identifier'] != '') & \
                                    (vehicle_feeds_engineOn['trip_identifier'].notna())].sort_values( \
                                            by = ['deviceCode', 'timestamp']).reset_index(drop = True)
    else:
        print("Vehicle is idle the whole time.")

def tripSummarizer(vehicle_feeds_with_trip_id):
    return vehicle_feeds_with_trip_id.groupby('trip_identifier').agg(
        start_time = ('timestamp', 'min'),
        time_diff = ('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        distance = ('distanceTravelled', 'sum')).reset_index().sort_values(by = ['trip_identifier'])
