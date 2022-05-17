import streamlit as st
import os
import torch
from PIL import Image
import cv2
from yolov5.detect import run 

#-------------------------------------------- 3. Layout -------------------------------------------------------------
st.set_page_config(
    page_title = 'HackBlue R&D Team',
    layout = 'wide'
)
layout = st.sidebar.columns([2, 3])

st.sidebar.title('HackBlue R&D Team')
st.sidebar.image('skunk.jpg', width=100)

#-------------------------------------- 4. User selection field -----------------------------------------------------

model = torch.load('best.pt')

choice = st.sidebar.radio("What's your favorite movie genre", ('Image', 'Video'))

file = st.sidebar.file_uploader("Choose a file", type=["png","jpg","jpeg","mp4","avi"])
st.markdown("<hr/>", unsafe_allow_html = True)


if file != None:
    with open(os.path.join("input/",file.name),"wb") as f:
        f.write((file).getbuffer())

            
    run(
    weights='best.pt',  # model.pt path(s)
    source=f"input/{file.name}",  # file/dir/URL/glob, 0 for webcam
    data='data.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=100,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project='runs/detect',  # save results to project/name
    name='hack_blue',  # save results to project/name
    exist_ok=True,  # existing project/name ok, do not increment
    line_thickness=2,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
)

    path = f'runs/detect/hack_blue/{file.name}'



    if choice == 'Image':
        row1, row2 = st.columns((3, 3))
            
    
        with row1:
            st.markdown(f'<p style="font-size: 25px;"><b>Original Image</b></p>', unsafe_allow_html=True)
            st.image(file)

        with row2:
            st.markdown(f'<p style="font-size: 25px;"><b>Processed Image</b></p>', unsafe_allow_html=True)
            st.image(path)

        st.markdown("<hr/>", unsafe_allow_html = True)
        
        row3, row4 = st.columns((3, 3))
        with row3:
            st.markdown(f'<p style="font-size: 25px;"><b>Some Statistics go Here</b></p>', unsafe_allow_html=True)
            #st.text(f'Number People: {total}')
            st.text('Something cool: 12')
        with row4:
            st.markdown(f'<p style="font-size: 25px;"><b>Other Statistics go Here</b></p>', unsafe_allow_html=True)
            st.text('Lets define this: 8')
            st.text('Some cool percentage: 55%')


    else:
        
        row1, row2 = st.columns((3, 3))
            

        with row1:
            st.markdown(f'<p style="font-size: 25px;"><b>Original Video</b></p>', unsafe_allow_html=True)
            st.video(file)

        with row2:
            st.markdown(f'<p style="font-size: 25px;"><b>Processed Video</b></p>', unsafe_allow_html=True)

            video_file = open(path, 'rb')
            video_s = video_file.read()
            st.video(video_s)

        st.markdown("<hr/>", unsafe_allow_html = True)
        row3, row4 = st.columns((3, 3))
        with row3:
            st.markdown(f'<p style="font-size: 25px;"><b>Some Statistics go Here</b></p>', unsafe_allow_html=True)
            #st.text(f'Number People: {total}')
            st.text('Something cool: 12')
        with row4:
            st.markdown(f'<p style="font-size: 25px;"><b>Other Statistics go Here</b></p>', unsafe_allow_html=True)
            st.text('Lets define this: 8')
            st.text('Some cool percentage: 55%')