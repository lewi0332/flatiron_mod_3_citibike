import pandas as pd
from mapbox import Geocoder

def set_geocoder(token):
    '''
    takes in your secret_token and returns your usable geocoder
    '''
    return Geocoder(access_token=token)

def get_neighborhood(row):
    '''
    uses the 'end station longitude' and 'end station latitude' of each row to find 
    the neighborhood from the mapbox. Mapbox returns a json file, which we filter 
    through in the second line.
    '''
    response = geocoder.reverse(lon=row['end station longitude'], lat=row['end station latitude'])
    return response.geojson()['features'][0]['context'][0]['text']


def change_birth_year(row,med_age_df):
    '''
    This function will be applied to each row, where ever there is a 1969 we will replace it 
    with the birth year from median_birthyear_no_69 which matches with our stop neighborhood
    '''
    if row['birth year'] == 1969:
        year = med_age_df[med_age_df['stop_nhbr']== row['stop_nhbr']]['birth year']
        return year.item()
    else:
        return row['birth year']
    
def add_neighborhoods(df,end_or_start):
    
    '''
    takes in the data frame and whether its for stop or start stations, then it will create a 
    DataFrame with all of the uniqe stops and thier neighborhood found using the get_neighborhood()
    function defined above.
    '''
    
    #create a dataframe of all unique stations
    station_names = df[end_or_start+' station name'].value_counts() 
    
    #create a list of the keys
    stn_names_li = list(station_names.keys()) 
        
    stops_df = pd.DataFrame(stn_names_li)
    stops_df.rename(columns={0:end_or_start+'_station_name'},inplace=True)

    if end_or_start == 'end':
        
        #concatinate the stop names with the latitude and longitude and drop unneeded columns
        stops_w_latlong = pd.concat([stops_df, df], axis=1, join='inner')
        stops_w_latlong = stops_w_latlong[['end station latitude','end station longitude',
                                           'start station name']] 
        
        stops_w_latlong['end_nhbr'] = stops_w_latlong.apply(lambda row:
                                                            get_neighborhood(row,'end'), axis=1)
       
        return stops_w_latlong
    
    elif end_or_start == 'start':

        stops_w_latlong = pd.concat([stops_df, df], axis=1, join='inner')
        stops_w_latlong = stops_w_latlong[['start station latitude','start station longitude',
                                           'start station name']]
        stops_w_latlong['start_nhbr'] = stops_w_latlong.apply(lambda row: 
                                                              get_neighborhood(row,'start'), axis=1)
        
        return stops_w_latlong