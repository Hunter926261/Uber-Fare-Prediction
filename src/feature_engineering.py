import numpy as np


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculates distance between two points using Haversine formula.
    """
    R = 6371  # Earth radius in KM
    
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def add_distance_feature(df):
    """
    Adds distance_km feature and removes raw coordinates.
    """
    df['distance_km'] = haversine_vectorized(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude']
    ).round(2)

    df = df.drop(columns=[
        'pickup_latitude',
        'pickup_longitude',
        'dropoff_latitude',
        'dropoff_longitude'
    ])

    return df
