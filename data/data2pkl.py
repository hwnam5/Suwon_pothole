import os
import cv2
import numpy as np
import pickle as pkl
import json

pothole_category = [1, 2, 6, 7, 8, 9]
batch_size = 20000
pothole_num = 0
not_pothole_num = 0

def data2pkl(file_name, data:dict):
    global pothole_num
    wo_jpg = file_name.split('.')[0]
    w_json = wo_jpg + ('.json')
    direction = file_name.split('_')[2] #direction
    
    if direction == 'F':
        image2matrix = cv2.imread(file_name, cv2.IMREAD_COLOR)
    else:
        image2matrix = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    resize_image2matrix = cv2.resize(image2matrix, (224, 224), interpolation=cv2.INTER_AREA)
    resize_image2matrix = resize_image2matrix / 255.0
    resize_image2matrix = resize_image2matrix - np.mean(resize_image2matrix)
    image2matrix_np = np.array(resize_image2matrix)
    
    with open(w_json, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    category_ids = set()
    for annotation in json_data['annotations']:
        category_ids.add(annotation['category_id'])
    
    pothole_sign = False
    for category_id in category_ids:
        if category_id in pothole_category:
            data[wo_jpg] = {
                #'file_name' : wo_jpg,
                'matrix' : image2matrix_np,
                'direction' : direction,
                'pothole' : 1
            }
            pothole_sign = True
            pothole_num += 1
            continue
    #if not pothole_sign:
        #data[wo_jpg] = {
            #'file_name' : wo_jpg,
            #'matrix' : image2matrix_np,
            #'direction' : direction,
            #'pothole' : 0
        #}
    #with open(f'pkl_files/data2pkl{pkl_num}.pkl', 'wb') as f:
        #pkl.dump(data, f)
    
    #os.remove(file_name)
    #os.remove(w_json)

data = {}
os.chdir('unzip/')
for i, file in enumerate(os.listdir()):
    if pothole_num % 5000 == 0 and pothole_num != 0:
        data = {}
        with open(f'pkl_files/pothole_224_{pothole_num//5000}.pkl', 'wb') as f:
            pkl.dump(data, f)
    if file.endswith('.jpg'):
        data2pkl(file, data)
    #if pothole_num % 5000 == 5000 - 1:
        #print(f'{i} images are put into tkl file')
        #with open(f'pkl_files/resize_224_{i//batch_size}.pkl', 'wb') as f:
            #pkl.dump(data, f)

#tkl 파일에 저장
#os.chdir('..')
with open('rgb_pothole.pkl', 'wb') as f:
    pkl.dump(data, f)