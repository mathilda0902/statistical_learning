import pandas as pd
import numpy as np
import json


'''read tripadvisor_json: 12774 hotels'''

'''codes for reading one .json, create one dataframe with user-item info'''
data = json.load(open('dataset/tripadvisor_json/72572.json'))
data.keys()
'''[u'Reviews', u'HotelInfo']'''

user = []
ratings = []
for review in data['Reviews']:
    user.append(review['Author'])
    ratings.append(review['Ratings']['Overall'])

user = pd.DataFrame(user, columns=['user'])
ratings = pd.DataFrame(ratings, columns=['ratings'])
user_mat = pd.concat([user, ratings], axis=1)

'''define read-in function'''
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
                            'value': [0]
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


        user_mat_sub = pd.concat([user, ratings, hotel_id, user_loc,
                                bus_service, front_desk, clean,
                                location, rooms, service, value], axis=1)
        user_mat = pd.concat([user_mat_sub, user_mat], axis=0)
    return user_mat

'''define read-in function: take two'''
def read_json(file):
    data = json.load(open(file))
    user_mat = []
    for hotel in data:
        user = []
        rating = []
        hotel_id = []
        for review in hotel['Reviews']:
            user.append(review['Author'])
            rating.append(review['Ratings']['Overall'])
            hotel_id.append(hotel['HotelInfo']['HotelID'])
        user = pd.DataFrame(user, columns=['user'])
        rating = pd.DataFrame(rating, columns=['ratings'])
        hotel_id = pd.DataFrame(hotel_id, columns=['hotel id'])
        user_mat_sub = pd.concat([user, rating, hotel_id], axis=1)
        user_mat.append(user_mat_sub)
    return user_mat


file2 = read_json('dataset/tripadvisor_json/72579.json')
file3 = read_json('dataset/tripadvisor_json/72586.json')
file23 = pd.concat([file2, file3], axis=0)

hotel = json.load(open('dataset/tripadvisor_json/80570.json'))

file_comp = read_json('dataset/tripadvisor_json/manifest.json')
file_comp2 = read_json('dataset/tripadvisor_json/sample/manifest2.json')

file_comp.to_csv('hotel_ratings.csv', encoding='utf-8', index=False)

hotel_ratings = pd.read_csv('dataset/tripadvisor_json/hotel_ratings.csv')




























if __name__ == '__main__':
    pass
