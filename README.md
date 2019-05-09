# Audio_Based_Language_Detection

Datasets used: https://drive.google.com/file/d/1M6j4ND9kTZITutPEFPqGotiT0biwEY19/view?usp=sharing, https://drive.google.com/drive/folders/1Fmh6iiipEGkpn1uaC9B2IdCTir3zbgQZ?fbclid=IwAR2Q88AfFFQ4iVvm2SLhGVjwXkie8Sw6IUlAsOWTooNUyZ8uPlGQFJ-UVck

model_train.py :
    - Line 219, 222 change between to_mfcc and to_chrome_cens to train with different types of features

    - Run by typing: python3 model_train.py test.csv modelfilename

    - Have the dataset in ./audio

    

test.csv: 
    - File needs to be generated by: 

        First line should have:
 
        - language_num,native_language
        
        Each of the following lines should be populated by:
        
        - filename,native_tongue    (filename shouldn't have the .wav)
            ex. bengali13,bengali 


demo_model.py:
    - Run file as: python3 demo_model.py <pathtomodel> <pathtoaudiofile> <feature>

    - Ensure that equivalent .txt file also present along with the model 
    ex. python3 demo_model.py ./models/modelmfcc.h5 ./audio/bengali10.wav mfcc
    (../models/modelmfcc.txt should be present)

    ex. python3 demo_model.py ./models/modelmfcc.h5 ./audio/mal1.wav chroma
    (../models/modelmfcc.txt should be present)
