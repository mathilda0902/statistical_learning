import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
import re


'''codes for reading one .json, create one dataframe with user-item info'''
data = json.load(open('dataset/tripadvisor_json/258990.json'))
data.keys()
'''[u'Reviews', u'HotelInfo']'''

'''function read json into user-item matrix'''
def read_json(file):
    data = json.load(open(file))
    user_mat = pd.DataFrame({'user': [0],
                            'ratings': [0],
                            'hotel id': [0],
                            'user location': [0],
                            'business service': [0],
                            'front desk': [0],
                            'cleanliness': [0],
                            'location': [0],
                            'rooms': [0],
                            'service': [0],
                            'value': [0],
                            'review date': [0],
                            'content': [0],
                            'title': [0],
                            'review id': [0]
                            })
    for hotel in data:
        user = []
        ratings = []
        hotel_id = []
        user_loc = []
        bus_service = []
        front_desk = []
        clean = []
        location = []
        rooms = []
        service = []
        value = []
        review_date = []
        content = []
        title = []
        review_id = []

        for review in hotel['Reviews']:
            user.append(review['Author'])
            ratings.append(review['Ratings']['Overall'])
            hotel_id.append(hotel['HotelInfo']['HotelID'])
            user_loc.append(review.get('AuthorLocation', float('NaN')))
            bus_service.append(review['Ratings'].get('Business service (e.g., internet access)', float('NaN')))
            front_desk.append(review['Ratings'].get('Check in / front desk', float('NaN')))
            clean.append(review['Ratings'].get('Cleanliness', float('NaN')))
            location.append(review['Ratings'].get('Location', float('NaN')))
            rooms.append(review['Ratings'].get('Rooms', float('NaN')))
            service.append(review['Ratings'].get('Service', float('NaN')))
            value.append(review['Ratings'].get('Value', float('NaN')))
            review_date.append(review.get('Date', float('NaN')))
            content.append(review.get('Content', float('NaN')))
            title.append(review.get('Title', float('NaN')))
            review_id.append(review.get('ReviewID', float('NaN')))

        user = pd.DataFrame(user, columns=['user'])
        ratings = pd.DataFrame(ratings, columns=['ratings'])
        hotel_id = pd.DataFrame(hotel_id, columns=['hotel id'])
        user_loc = pd.DataFrame(user_loc, columns=['user location'])
        bus_service = pd.DataFrame(bus_service, columns=['business service'])
        front_desk = pd.DataFrame(front_desk, columns=['front desk'])
        clean = pd.DataFrame(clean, columns=['cleanliness'])
        location = pd.DataFrame(location, columns=['location'])
        rooms = pd.DataFrame(rooms, columns=['rooms'])
        service = pd.DataFrame(service, columns=['service'])
        value = pd.DataFrame(value, columns=['value'])
        review_date = pd.DataFrame(review_date, columns=['review date'])
        content = pd.DataFrame(content, columns=['content'])
        title = pd.DataFrame(title, columns=['title'])
        review_id = pd.DataFrame(review_id, columns=['review id'])


        user_mat_sub = pd.concat([user, ratings, hotel_id, user_loc,
                                bus_service, front_desk, clean,
                                location, rooms, service, value,
                                review_date, content, title, review_id], axis=1)
        user_mat = pd.concat([user_mat_sub, user_mat], axis=0)
    return user_mat


'''smaller sample for testing read_json function'''
file_comp2 = read_json('dataset/tripadvisor_json/sample/manifest2.json')

'''entire json'''
file_comp = read_json('dataset/tripadvisor_json/manifest.json')

'''saving as csv'''
file_comp.to_csv('hotel_ratings.csv', encoding='utf-8', index=False)
'''reading to pandas'''
hotel_ratings = pd.read_csv('dataset/hotel_ratings.csv')



'''function to extract hotel info:'''
def read_hotel(file):
    data = json.load(open(file))

    name = []
    hotel_id = []
    city = []
    state = []
    price_low = []
    price_high = []
    for hotel in data:
        if 'Address' in hotel['HotelInfo'].keys():
            soup1 = BeautifulSoup(hotel['HotelInfo']['Address'], 'html.parser')
            text1 = soup1.get_text().strip()
            output1 = re.split(',', text1)
            state_name = [e.strip() for e in output1][-1]
            city_name = output1[-2].lstrip()
        if 'Price' in hotel['HotelInfo'].keys():
            soup2 = BeautifulSoup(hotel['HotelInfo']['Price'], 'html.parser')
            text2 = soup2.get_text()
            output2 = re.split('-', text2)
            price_range = [e.strip(' |* ') for e in output2]

        name.append(hotel['HotelInfo'].get('Name', float('NaN')))
        hotel_id.append(hotel['HotelInfo'].get('HotelID', float('NaN')))
        city.append(city_name)
        state.append(state_name)
        try:
            ph = price_range[1]
        except IndexError:
            ph = 'null'
        price_low.append(price_range[0])
        price_high.append(ph)

    name = pd.DataFrame(name, columns=['hotel name'])
    hotel_id = pd.DataFrame(hotel_id, columns=['hotel id'])
    city = pd.DataFrame(city, columns=['city'])
    state = pd.DataFrame(state, columns=['state'])
    price_low = pd.DataFrame(price_low, columns=['low price'])
    price_high = pd.DataFrame(price_high, columns=['high price'])

    hotel_mat = pd.concat([name, hotel_id, city, state, price_low, price_high], axis=1)
    return hotel_mat

'''testing read_hotel function with sample data:'''
file_comp2 = read_hotel('dataset/tripadvisor_json/sample/manifest2.json')
file_comp2.to_csv('dataset/tripadvisor_json/sample_hotel.csv', encoding='utf-8', index=False)
sample_hotel = pd.read_csv('dataset/tripadvisor_json/sample_hotel.csv')

'''read entire hotel info:'''
file_compiled = read_hotel('dataset/tripadvisor_json/manifest.json')

'''saving as csv'''
file_compiled.to_csv('dataset/tripadvisor_json/hotel_info.csv', encoding='utf-8', index=False)
'''reading to pandas'''
hotel_info = pd.read_csv('dataset/hotel_info.csv')







if __name__ == '__main__':
    pass
