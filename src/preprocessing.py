import pandas as pd

def clean_data(df):
    # Removes missing values and unwanted columns.
    
    df = df.dropna()
    df = df.drop(columns=['key', 'Unnamed: 0'], errors='ignore')
    return df


def extract_datetime_features(df):
    
    # Extract year, month, day, hour from pickup_datetime.
    
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['hour'] = df['pickup_datetime'].dt.hour

    df = df.drop(columns=['pickup_datetime'])
    return df


def validate_coordinates(df):
    
    # Removes invalid latitude and longitude values.
    
    df = df[
        (df['pickup_latitude'].between(-90, 90)) &
        (df['dropoff_latitude'].between(-90, 90)) &
        (df['pickup_longitude'].between(-180, 180)) &
        (df['dropoff_longitude'].between(-180, 180))
    ]

    df = df.dropna(subset=[
        'pickup_latitude',
        'pickup_longitude',
        'dropoff_latitude',
        'dropoff_longitude'
    ])

    return df
