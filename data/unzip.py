import os
import zipfile as zp

def unzip(dir):
    for file in os.listdir(dir):
        if file.endswith('.zip'):
            zip_fol = file
            os.chdir(dir)
            zp.ZipFile(zip_fol).extractall(dir)
            print(f'{zip_fol} has been unzipped')
    
    os.chdir('../../..')

zip_fol_list = ['data/image_data/Training/image', 
                'data/image_data/Training/label', 
                'data/image_data/Validation/image', 
                'data/image_data/Validation/label']

for zip_fol in zip_fol_list:
    unzip(zip_fol)

