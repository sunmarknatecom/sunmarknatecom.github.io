```python
# STEP 1: Read files
# STEP 2: Complete the dataframe
# STEP 3: Get the pixel from the files
# STEP 4: Transform to input data
#        1) sum model
#        2) individual model
# STEP 5: Construct the model (pytorch)
# STEP 6: Training by pytorch
# STEP 7: Test
# STEP 8: Grad-CAM


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import matplotlib.animation as animation

# wanted variables
# col_name = ['BraTS21ID',
            # 'seq1', 'num1', 'size1', 'ori1', 'pos1', 'csize1',
            # 'seq2', 'num2', 'size2', 'ori2', 'pos2', 'csize2',
            # 'seq3', 'num3', 'size3', 'ori3', 'pos3', 'csize3',
            # 'seq4', 'num4', 'size4', 'ori4', 'pos4', 'csize4',
            # ]
# explanation:
#   num is file numbers in the seq1 of BraTS21ID
#   size is the representative image size in the all file of seq
#   ori is the orientation of all files in the seq directory
#   pos is the position of all files in the seq directory
#   csize is the size of cropped images in the seq directory

def normalization(ds):
    '''
    Input data : numpy array
    Output data: numpy array with max 1 and min 0
    Error : max of array is zero
    '''
    try:
        MAX_val = np.max(ds)
        MIN_val = np.min(ds)
        NUME = ds - MINV
        DENO = MAX_val - MIN_val
        ret_ds = NUME / DENO
        return ret_ds
    except:
        print("Array is not available, max is zero.")

def animation_transform(root, idx, seq):
    '''
    make the animation using matplotlib
    ex) list of multiple images
    '''
    ROOT_PATH = f"./train/{idx}/{seq}"
    IMG_LIST = []
    RAW_FL =[int(name.split("-")[1].split(".")[0]) for name in os.listdir(ROOT_PATH)]
    FILE_LIST = sorted(RAW_FL)
    NEW_FILE_LIST = [f'Image-{name}.dcm' for name in FILE_LIST]
    for f_name in NEW_FILE_LIST:
        f = pydicom.dcmread(f'{ROOT_PATH}/{f_name}')
        ds = f.pixel_array
        IMG_LIST.append(ds)
    MAX_VAL = np.max(IMG_LIST)
    MIN_VAL = np.min(IMG_LIST)
    DIVUP = IMG_LIST - MIN_VAL
    DIVDW = MAX_VAL - MIN_VAL
    NOR_IMG = DIVUP / DIVDW
    return NOR_IMG

def show_animation(list):
    fig, ax = plt.subplots()
    ims = []
    for i in list:
        im = ax.imshow(i, interpolation='bicubic', cmap='gray', vmax=1, vmin=0)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.show()

def check_list(root='.'):
    '''
    Check the list, sequence between csv and real data
    '''

    # List of csv file
    CSV_DF = pd.read_csv(f'{root}/train_labels.csv') # original DF
    IDX_LIST_CSV = CSV_DF['BraTS21ID'].tolist()
    strIDX_LIST_CSV = [str(str_idx).zfill(5) for str_idx in IDX_LIST_CSV]

    # List from directory
    IDX_LIST_DIR = os.listdir(f'{root}/train/') # ['00000', '00002',...

    ### result
    if sorted(strIDX_LIST_CSV) == sorted(IDX_LIST_DIR):
        print("All idx have same sequences, no problem")
    else:
        print("Something is wrong.!!!")

#------------------------------------------------------------------------------

# STEP 2
# Makeing the basic dataframe
#------------------------------------------------------------------------------
ROOT_PATH = '.'
tmp_df = []
tmp_idx_list = []
NA_IMG_LIST = [] # not available object list
SEQ_LIST = ['FLAIR', 'T1w', 'T1wCE', 'T2w']

for idx in IDX_LIST_DIR:
    tmp_idx_list.append(idx) # ex) ['00000','00002', ...
    # check the 
    # ``````
    tmp_seq_list = []
    for seq_dir in SEQ_LIST: # ex) ['FLAIR', 'T1w', 'T1wCE', 'T2w']        
        FULL_PATH = f'{ROOT_PATH}/train/{idx}/{seq_dir}'
        name_seq = f'n(seq_dir)'
        nSeq = len(os.listdir(FULL_PATH))       # var: num1
        RE_ARR_LIST = sorted(int(num.split("-")[1].split(".")[0]) for num in os.listdir(FULL_PATH))
        ITR_LIST = [f'Image-{elem}.dcm' for elem in RE_ARR_LIST]
        for idx_obj in ITR_LIST: # ["Image-1.dcm", ...
            tmp_obj_list = []
            input_path = f'{FULL_PATH}/{idx_obj}'
            f = pydicom.dcmread(input_path)
            ds = f.pixel_array
            ht_img = f.Rows                     # var: 
            wd_img = f.Columns                  # var 
            try:
                # print(idx, seq_dir, idx_obj, end=' ')
                ser_f = f.SliceLocation
            except AttributeError:
                print("error f.SL = ", f'{idx}/{seq_dir}/{idx_obj}')
                NA_IMG_LIST.append(['NA f.Sl', idx, seq_dir, idx_obj])
                pass
            ori_f  = f.ImageOrientationPatient  # var
            pos_f  = f.PatientPosition          # var
            tmp_ser = [idx, seq_dir, idx_obj, ht_img, wd_img, ser_f, ori_f, pos_f]
            tmp_df.append(tmp_ser)
            if f.PatientID == idx:
                if f.SeriesDescription == seq_dir:
                    pass
                else:
                    print("Not equal to Series", idx_obj)
                    NA_IMG_LIST.append(idx_obj)
                    pass
            else:
                print("Not equal to BraTS21ID ", idx_obj)
            if np.max(ds) != 0:
                tmp_ser = [idx, seq_dir, idx_obj, ht_img, wd_img, ser_f, ori_f, pos_f]
            else:
                tmp_ser = ['Not img',idx, seq_dir, idx_obj, ht_img, wd_img, ser_f, ori_f, pos_f]
                NA_IMG_LIST.append(tmp_ser)
                pass
            del input_path, f, ds
            del ht_img, wd_img, ori_f, pos_f
#------------------------------------------------------------------------------
# if sorted(tmp_idx_list) == sorted(IDX_LIST_CSV):
#     print("All images were analyzed")
# else:
#     print("Not be analyzed, in ")
#     for elem  in tmp_seq_list:
#         print(elem, end=', ')
#     print(".")
WHOLE_DF = pd.DataFrame(tmp_df)

WHOLE_DF.to_csv('TMP_DF.csv', mode='w')

IDX = 00000
SEQ = "FLAIR"

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

# Basic structure code to get the files from directory

import os
import pydicom

ROOT = './train'
ITR_IDX = sorted(os.listdir(ROOT)):

for idx in ITR_IDX:
    PATH_IDX = f'{ROOT}/{idx}'
    ITR_SEQ = sorted(os.listdir(PATH_IDX))

    # current path) ex) ./train/00000

    # -------------------------------------------------------------------------
    # Insert code

    # -------------------------------------------------------------------------

    for seq_idx in ITR_SEQ:
        PATH_SEQ = f'{PATH_IDX}/{seq_idx}'
        ITR_OBJ = os.listdir(PATH_SEQ)

        # sorted list of image object
        SRT_OBJ = sorted(int(num_split("-")[1].split(".")[0]) for num in ITR_OBJ)

        # current path) ex) ./train/00000/FLAIR
        # ---------------------------------------------------------------------
        # Insert code

        # ---------------------------------------------------------------------

        for img_idx in SRT_OBJ:
            FULL_PATH = f'{PATH_SEQ}/{img_idx}'

            # FULL_PATH ex) ./train/00000/FLAIR/Image-1.dcm
            # ---------------------------------------------------------------------
            # Insert code

            # read the dicom file
            obj_img = pydicom.dcmread(FULL_PATH)
            
            
```
