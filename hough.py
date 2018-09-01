import cv2
import numpy as np
import imutils
import pandas as pd

RESIZE_WIDTH = 500
SAME_ROW_Y_AXIS_OFFSET = 15


def extract_ROI_value(data, MEAN_R, img6, circle_mask):
    #TODO : exception
    answer_circle_ROI = img6[data[1]-MEAN_R:data[1]+MEAN_R,
                             data[0]-MEAN_R:data[0]+MEAN_R,
                             :].copy()
    answer_circle_ROI = cv2.bitwise_and(answer_circle_ROI, answer_circle_ROI, mask=circle_mask)
    return np.sum(answer_circle_ROI)

def min_ROI_data(dataset):
    min_ROI = dataset[0]['roi']
    answer_data = dataset[0]['data']
    answer_option = 1
    for option, data in enumerate(dataset):
        if data['roi'] < min_ROI:
            min_ROI = data['roi']
            answer_data = data['data']
            answer_option = option + 1
    return (answer_option, answer_data)


def group_circle_by_y(df, option_number, r, img6, circle_mask):
    # collect circle data
    # according to 'y', grouping circle
    group = dict()
    grouped = False
    for index, data in df.iterrows():
        for key in group.keys():
            int_key = int(key)
            if int_key - SAME_ROW_Y_AXIS_OFFSET < data['y'] and data['y'] < int_key + SAME_ROW_Y_AXIS_OFFSET:
                group[key].append(dict(data=data, roi=extract_ROI_value(data, r, img6, circle_mask)))
                grouped = True
                break
        if not grouped:
            group[str(data['y'])] = [dict(data=data, roi=extract_ROI_value(data, r, img6, circle_mask))]
        grouped = False

    keys = list(group.keys())
    for key in keys:
        # check error detect
        if len(group[key]) <= 1:
            group.pop(key)
            continue
        #TODO
        #if len(group[key]) > option_number:
        #    return []
        group[key] = sorted(group[key], key=lambda x: x['data']['x'])
    return group

def collect_answer(group, question_number):
    # min_ROI_data would return a tuple! (option_order, data)
    result = [min_ROI_data(group[key]) for key in group.keys()]
    result = sorted(result, key=lambda x: x[1]['y'])

    #for i in result:
    #    print('*'*50)
    #    print(i)

    result = [choose[0] for choose in result]
    #TODO : comment
    #if len(result) != question_number:
    #   return []
    return result

def show_image(img6, origin_img3):
    cv2.imshow('detected circles',img6)
    cv2.imshow('original pic', origin_img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_para(filename, para1, para2):
    question_number = 5
    option_number = 5


    #filename = input('name : ')
    #origin_img3 = cv2.imread(filename + '.jpeg')
    origin_img3 = cv2.imread(filename)

    origin_img3 = imutils.resize(origin_img3, width=RESIZE_WIDTH)
    #img3 = cv2.GaussianBlur(origin_img3,(3, 3), 1)
    img3 = cv2.medianBlur(origin_img3, 3)
    img3 = cv2.GaussianBlur(img3,(3, 3), 1)
    #img3 = cv2.blur(origin_img3, (3,3))
    #img3 = origin_img3
    #cv2.imshow('blur', img3)
    img1 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(img1,cv2.HOUGH_GRADIENT,1,20,param1=para1,param2=para2,minRadius=6,maxRadius=16)

    circles = np.uint16(np.around(circles))

    MEAN_R = np.uint16(np.ceil(np.mean(circles, axis=1)[0][2]))

    img6 = origin_img3.copy()
    circle_mask = np.zeros((MEAN_R*2, MEAN_R*2), dtype=np.uint8)
    cv2.circle(img=circle_mask,
               center=(MEAN_R, MEAN_R),
               radius=MEAN_R,
               color=(255, 255, 255),
               thickness=cv2.FILLED
              )

    EROSION_KERNEL = np.ones((2,2), np.uint8)

    for i in circles[0, :]:
        answer_circle_ROI = img6[i[1]-MEAN_R:i[1]+MEAN_R,
                                 i[0]-MEAN_R:i[0]+MEAN_R,
                                 :]
        answer_circle_ROI = cv2.bitwise_and(answer_circle_ROI, answer_circle_ROI, mask=circle_mask)
        answer_circle_ROI = cv2.adaptiveThreshold(cv2.cvtColor(answer_circle_ROI, cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,1)


        answer_circle_ROI = cv2.erode(answer_circle_ROI, EROSION_KERNEL, iterations=1)

        img6[i[1]-MEAN_R:i[1]+MEAN_R, i[0]-MEAN_R:i[0]+MEAN_R, :] = cv2.cvtColor(answer_circle_ROI, cv2.COLOR_GRAY2BGR)

        #cv2.circle(img6,(i[0],i[1]),MEAN_R,(0,255,0),2)
        #cv2.circle(img6,(i[0],i[1]),2,(0,0,255),1)

    dataframe = pd.DataFrame(circles.reshape(-1, 3), columns=['x', 'y', 'r'])
    group = group_circle_by_y(dataframe, option_number,MEAN_R, img6, circle_mask)
    res = collect_answer(group, question_number)
    return res

# 30 25
ind = 0
ans = [2, 3, 1 , 4, 5, 2, 5, 1]
all_file = ['d/1.jpeg', 'd/2.jpeg', 'd/3.jpeg', 'd/4.jpeg', 'd/5.jpeg', 'd/6.jpeg', 'd/7.jpeg', 'd/8.jpeg']
args1 = [98, 182, 266, 350]
args2 = [18]
for arg1 in args1:#100~350/18
    for arg2 in args2:
        count = 0
        print('*'*50)
        print(arg1, arg2)
        for filename in all_file:
            try:
                result = test_para(filename, arg1, arg2)
            except:
                continue
            if result == ans:
                print(filename.split('/')[1].split('.')[0], end=" ")
                count = count + 1
            ind = ind + 1
        print()
        if count >= 6:
            print(arg1, arg2)
