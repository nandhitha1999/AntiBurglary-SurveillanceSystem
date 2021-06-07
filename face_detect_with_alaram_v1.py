from tensorflow.keras.models import Model
import os
import cv2
import tensorflow as tf
from keras import backend as K
import numpy as np
from PIL import Image
import winsound
from tensorflow.keras.layers import Input
from utils import letterbox_image, tf_eval, preprocess_input, retrieve_tf_detections, draw_boxes
from tfv3 import tfv3_main
from twilio.rest import Client
from playsound import playsound
from face_reg import check_img

def detectObjectsFromImage(
    input_image='',
    output_image_path='',
    input_type='file',
    output_type='file',
    extract_detected_objects=False,
    minimum_percentage_probability=50,
    display_percentage_probability=True,
    display_object_name=True,
    display_box=True,
    thread_safe=False,
    custom_objects=None,
    ):
    if __modelLoaded == False:
        raise ValueError('You must call the loadModel() function before making object detection.'
                         )
    elif __modelLoaded == True:
        try:
            model_detections = list()
            detections = list()
            image_copy = None

            detected_objects_image_array = []
            min_probability = minimum_percentage_probability / 100

            if input_type == 'file':
                input_image = cv2.imread(input_image)
            elif input_type == 'array':
                input_image = np.array(input_image)

            detected_copy = input_image
            image_copy = input_image

            if True:
                (image_h, image_w, _) = detected_copy.shape
                detected_copy = preprocess_input(detected_copy,__tf_model_image_size)

                model = __model_collection[0]
                tf_result = model.predict(detected_copy)
                model_detections = retrieve_tf_detections(
                    tf_result,
                    __tf_anchors,
                    min_probability,
                    __nms_thresh,
                    __tf_model_image_size,
                    (image_w, image_h),
                    numbers_to_names,
                    )
                jj = 0
                kok = ''
                for i in model_detections:
                    print(i)
                    print(i['name'])
                    if i['name'] == 'person':
                        kok = check_img( input_image )
                        if kok != '':
                            print(kok)
                            continue
                    
                        jj = jj + 1
                        account_sid = 'AC861a1918479db70f624f50d85ba34385'
                        auth_token = '24a1bd114ac0125b1448a446540a0763'
                        client = Client(account_sid, auth_token)
                        message = client.messages.create( 
                              from_='+19708343329',  
                              body='Person Detected',      
                              to='+919344485320' )
                        playsound(os.path.join(os.getcwd(),"alarm.mp3"))
                        print('person***')
                print(jj)

            counting = 0
            objects_dir = output_image_path + '-objects'

            for detection in model_detections:
                counting += 1
                if kok == '':
                    label = detection['name']
                else:
                    label = kok
                percentage_probability = detection['percentage_probability']
                box_points = detection['box_points']

                if custom_objects is not None:
                    if custom_objects[label] != 'valid':
                        continue

                detections.append(detection)

                if display_object_name == False:
                    label = None

                if display_percentage_probability == False:
                    percentage_probability = None

                image_copy = draw_boxes(
                    image_copy,
                    box_points,
                    display_box,
                    label,
                    percentage_probability,
                    __box_color,
                    )

                if extract_detected_objects == True:
                    splitted_copy = image_copy.copy()[box_points[1]:
                            box_points[3], box_points[0]:box_points[2]]
                    if output_type == 'file':
                        if os.path.exists(objects_dir) == False:
                            os.mkdir(objects_dir)
                        splitted_image_path = os.path.join(objects_dir,
                                detection['name'] + '-' + str(counting)
                                + '.jpg')
                        cv2.imwrite(splitted_image_path, splitted_copy)
                        detected_objects_image_array.append(splitted_image_path)
                    elif output_type == 'array':
                        detected_objects_image_array.append(splitted_copy)

            if output_type == 'file':
                cv2.imwrite(output_image_path, image_copy)

            if extract_detected_objects == True:
                if output_type == 'file':
                    return (detections, detected_objects_image_array)
                elif output_type == 'array':
                    return (image_copy, detections,
                            detected_objects_image_array)
            else:

                if output_type == 'file':
                    return detections
                elif output_type == 'array':
                    return (image_copy, detections)
        except Exception as e:
            print(e)
            raise ValueError('Ensure you specified correct input image, input type, output type and/or output image path '
                             )


# get the current execution path
execution_path = os.getcwd()
# get the camera input
camera = cv2.VideoCapture(0)
input_video = camera
# output video path
output_video_filepath = os.path.join(execution_path,"camera_detected_video") + '.avi'
# get the camera attributes
frame_width = int(input_video.get(3))
frame_height = int(input_video.get(4))
# set the output video attributes
frames_per_second = 20 #default
output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                               frames_per_second,
                                               (frame_width, frame_height))
# show the processing log
log_progress = True

# detection speed
detection_speed = 'normal'
if detection_speed == 'normal':
    __tf_model_image_size = (416, 416)
elif detection_speed == 'fast':
    __tf_model_image_size = (320, 320)
elif detection_speed == 'faster':
    __tf_model_image_size = (208, 208)
elif detection_speed == 'fastest':
    __tf_model_image_size = (128, 128)
elif detection_speed == 'flash':
    __tf_model_image_size = (96, 96)

# load model
__model_collection = []
__tf_anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
input_image = Input(shape=(None, None, 3))
numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                                 6: 'train',
                                 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                                 12: 'parking meter',
                                 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                                 20: 'elephant',
                                 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                                 27: 'tie',
                                 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                                 33: 'kite',
                                 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                                 38: 'tennis racket',
                                 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                                 45: 'bowl',
                                 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                                 52: 'hot dog',
                                 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                                 59: 'bed',
                                 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                                 66: 'keyboard',
                                 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                                 72: 'refrigerator',
                                 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                                 78: 'hair dryer',
                                 79: 'toothbrush'}
model = tfv3_main(input_image, len(__tf_anchors), len(numbers_to_names.keys()))
model.load_weights(os.path.join(execution_path,"tf.h5"))
__model_collection.append(model)
__modelLoaded = True

minimum_percentage_probability = 30
frame_detection_interval = 1
__tf_iou = 0.45
__tf_score = 0.1
__nms_thresh = 0.45
__box_color = (112, 19, 24)
save_detected_video = True

counting = 0
output_frames_dict = {}
output_frames_count_dict = {}
detection_timeout_count = 0
video_frames_count = 0
detection_timeout = None
per_frame_function = None
per_second_function = None
per_minute_function = None
video_complete_function = None
return_detected_frame = None
detection_timeout = None
thread_safe = None
while input_video.isOpened():
    (ret, frame) = input_video.read()
    if ret == True:
        video_frames_count += 1
        if detection_timeout != None:
            if video_frames_count % frames_per_second == 0:
                detection_timeout_count += 1
            if detection_timeout_count >= detection_timeout:
                break
        if True:
            if True:
                if True:
                    output_objects_array = []

                    counting += 1

                    if log_progress == True:
                        print ('Processing Frame : ', str(counting))

                    detected_copy = frame.copy()

                    check_frame_interval = counting % frame_detection_interval

                    if counting == 1 or check_frame_interval == 0:
                        try:
                            (detected_copy, output_objects_array) = detectObjectsFromImage(
                                input_image=frame,
                                input_type='array',
                                output_type='array',
                                minimum_percentage_probability=minimum_percentage_probability,
                                display_percentage_probability=True,
                                display_object_name=True,
                                display_box=True,
                                custom_objects=None,
                                )
                        except Exception as e:
                            print(e)
                            None

                    output_frames_dict[counting] = output_objects_array

                    output_objects_count = {}
                    for eachItem in output_objects_array:
                        eachItemName = eachItem['name']
                        try:
                            output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                        except:
                            output_objects_count[eachItemName] = 1

                    output_frames_count_dict[counting] = output_objects_count

                    if save_detected_video == True:
                        output_video.write(detected_copy)

                    if counting == 1 or check_frame_interval == 0:
                        if per_frame_function != None:
                            if return_detected_frame == True:
                                per_frame_function(counting,
                                        output_objects_array,
                                        output_objects_count,
                                        detected_copy)
                            elif return_detected_frame == False:
                                per_frame_function(counting,
                                        output_objects_array,
                                        output_objects_count)

                    if per_second_function != None:
                        if counting != 1 and counting \
                            % frames_per_second == 0:

                            this_second_output_object_array = []
                            this_second_counting_array = []
                            this_second_counting = {}

                            for aa in range(counting):
                                if aa >= counting - frames_per_second:
                                    this_second_output_object_array.append(output_frames_dict[aa + 1])
                                    this_second_counting_array.append(output_frames_count_dict[aa + 1])

                            for eachCountingDict in this_second_counting_array:
                                for eachItem in eachCountingDict:
                                    try:
                                        this_second_counting[eachItem] = this_second_counting[eachItem] + eachCountingDict[eachItem]
                                    except:
                                        this_second_counting[eachItem] = eachCountingDict[eachItem]

                            for eachCountingItem in this_second_counting:
                                this_second_counting[eachCountingItem] = int(this_second_counting[eachCountingItem] / frames_per_second)

                            if return_detected_frame == True:
                                per_second_function(int(counting / frames_per_second),
                                        this_second_output_object_array,
                                        this_second_counting_array,
                                        this_second_counting,
                                        detected_copy)
                            elif return_detected_frame == False:

                                per_second_function(int(counting / frames_per_second),
                                        this_second_output_object_array,
                                        this_second_counting_array,
                                        this_second_counting)

                    if per_minute_function != None:

                        if counting != 1 and counting % (frames_per_second * 60) == 0:

                            this_minute_output_object_array = []
                            this_minute_counting_array = []
                            this_minute_counting = {}

                            for aa in range(counting):
                                if aa >= counting - frames_per_second * 60:
                                    this_minute_output_object_array.append(output_frames_dict[aa + 1])
                                    this_minute_counting_array.append(output_frames_count_dict[aa + 1])

                            for eachCountingDict in this_minute_counting_array:
                                for eachItem in eachCountingDict:
                                    try:
                                        this_minute_counting[eachItem] = this_minute_counting[eachItem] + eachCountingDict[eachItem]
                                    except:
                                        this_minute_counting[eachItem] = eachCountingDict[eachItem]

                            for eachCountingItem in this_minute_counting:
                                this_minute_counting[eachCountingItem] = int(this_minute_counting[eachCountingItem] / (frames_per_second * 60))

                            if return_detected_frame == True:
                                per_minute_function(int(counting / (frames_per_second * 60)),
                                        this_minute_output_object_array,
                                        this_minute_counting_array,
                                        this_minute_counting,
                                        detected_copy)
                            elif return_detected_frame == False:

                                per_minute_function(int(counting / (frames_per_second * 60)),
                                        this_minute_output_object_array,
                                        this_minute_counting_array,
                                        this_minute_counting)
                else:

                    break

if video_complete_function != None:
    this_video_output_object_array = []
    this_video_counting_array = []
    this_video_counting = {}
    for aa in range(counting):
        this_video_output_object_array.append(output_frames_dict[aa + 1])
        this_video_counting_array.append(output_frames_count_dict[aa + 1])
    for eachCountingDict in this_video_counting_array:
        for eachItem in eachCountingDict:
            try:
                this_video_counting[eachItem] = this_video_counting[eachItem] + eachCountingDict[eachItem]
            except:
                this_video_counting[eachItem] = eachCountingDict[eachItem]

    for eachCountingItem in this_video_counting:
        this_video_counting[eachCountingItem] = int(this_video_counting[eachCountingItem] / counting)
    video_complete_function(this_video_output_object_array,
                            this_video_counting_array,
                            this_video_counting)
input_video.release()
output_video.release()
