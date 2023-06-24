import pandas as pd
import re
import numpy as np
from datetime import datetime

def prepare_data(df):
    pd.set_option('mode.chained_assignment', None) #avoid warnings
    
    df['City'] = df['City'].str.strip() #removing duplicate cities because of spaces
    df['price'] = df['price'].apply(lambda x: re.sub(',','', str(x))).str.extract(r'(\d+)') #extracting price
    df['Area'] = df['Area'].apply(lambda x: re.sub('[^0-9]','',str(x))) #extracting area
    df['room_number'] = df['room_number'].apply(lambda x: re.sub('[^0-9\.]+','',str(x))) #extracting number of rooms

    df[['price','Area','room_number']] = df[['price','Area','room_number']].applymap(pd.to_numeric, errors='coerce') #transforming columns to float

    df = df.dropna(subset='price') #removing rows that do not contain price

    cols = ['City', 'type', 'Street', 
         'floor_out_of', 'hasElevator ','city_area',
       'hasParking ', 'hasBars ', 'hasStorage ',
       'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ',
        'furniture ', 'description '] #preparing columns to remove unnecessary characters

    df[cols] = df[cols].applymap(lambda x: re.sub('[^א-ת0-9a-zA-Z ]','',str(x))) #removing unnecessary characters

    df['floor_out_of_1'] = df['floor_out_of'].str.replace('קרקע', '0').str.replace('-מרתף', '1') #transforming minus floor to -1 and ground floor to 0

    df['floor'] = df['floor_out_of_1'].str.extract(r'(\d+)') #extracting floor number
    df['total_floors'] = df['floor_out_of_1'].str.extract(r'מתוך (\d+)') #extracting number of floors in the building

    df.loc[df['floor_out_of'].str.contains('קומת מרתף'), 'floor'] = -1 #assign -1 to minus floors

    #transforming columns to int
    df[['total_floors', 'floor', 'number_in_street','num_of_images']] = df[['total_floors', 'floor', 'number_in_street','num_of_images']].applymap(pd.to_numeric, errors='coerce').astype('Int64')

    df.loc[df['floor'] > df['total_floors'], 'total_floors'] = np.NaN #removing total of floors if the floor is bigger than the floors total

    df.drop('floor_out_of_1', axis=1, inplace=True) #removing helping column

    #Creating and applying function to transform entrance date column to categorical column
    def classify_duration(value):
        today = datetime.today()
        if isinstance(value, datetime):
            months_diff = (value.year - today.year) * 12 + (value.month - today.month)
            if months_diff < 6:
                return 'less_than_6_months'
            elif 6 <= months_diff < 12:
                return 'months_6_12'
            else:
                return 'above_year'
        else:
            return value
    
    df['entranceDate '] = df['entranceDate '].replace('גמיש','flexible').replace('גמיש ','flexible').replace('מיידי', 'less_than_6_months').replace('לא צויין', 'not_defined')
    df['entranceDate '] = df['entranceDate '].apply(classify_duration)

    df['city_area'] = df['city_area'].replace(['nan','2003','2005'],np.NaN) #removing specific odd values
    df['city_area'] = df['city_area'].str.strip() #removing duplicate city areas because of spaces
    
    df.drop_duplicates(inplace=True) #removing duplicated rows in the dataframe

    df['condition '] = df['condition '].replace([False, np.NaN], 'לא צויין') #standardize odd values to known ones

    #Creating binary columns by uniforming the data
    df = df.replace({'True': 1, 'False': 0})
    okay = [1,'yes','יש סורגים', 'יש', 'כן', 'יש מעלית', 'יש חניה', 'יש חנייה', 'יש סרוגים', 'יש מחסן', 'יש מיזוג אוויר', 'יש מיזוג אויר', 'יש מרפסת', 'יש ממד', 'נגיש', 'נגיש לנכים']
    notokay = [0, 'no', 'אין סורגים', 'אין', 'לא', 'אין מעלית', 'אין חניה', 'אין סרוגים', 'אין מיזוג אויר', 'אין מחסן', 'אין מרפסת', 'אין ממד', 'לא נגיש לנכים', 'לא נגיש', 'nan']

    df['hasElevator '] = df['hasElevator '].replace(okay, 1).replace(notokay, 0)
    df['hasParking '] = df['hasParking '].replace(okay, 1).replace(notokay, 0)
    df['hasBars '] = df['hasBars '].replace(okay, 1).replace(notokay, 0)
    df['hasStorage '] = df['hasStorage '].replace(okay, 1).replace(notokay, 0)
    df['hasAirCondition '] = df['hasAirCondition '].replace(okay, 1).replace(notokay, 0)
    df['hasBalcony '] = df['hasBalcony '].replace(okay, 1).replace(notokay, 0)
    df['hasMamad '] = df['hasMamad '].replace(okay, 1).replace(notokay, 0)
    df['handicapFriendly '] = df['handicapFriendly '].replace(okay, 1).replace(notokay, 0)

    return df
