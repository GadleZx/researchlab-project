from src.vitpose_infer import VitInference
import cv2 

vid = cv2.VideoCapture('../../data/data_002.mp4')
model = VitInference('models/vitpose-b-multi-coco.pth',\
            yolo_path='yolov5n.engine',tensorrt=False)
frame_counter =0

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('data_002_pose.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))
 
while True:
    ret,frame = vid.read()
    if ret:
        # print(ret)
        pts,tids,bboxes,drawn_frame,orig_frame= model.inference(frame,frame_counter)
        cv2.imshow("Video",drawn_frame)
        cv2.waitKey(1)
        frame_counter+=1

        # Write the frame into the file 'output.avi'
        out.write(drawn_frame)

    else:
        break
vid.release()
out.release()
# writer.release()  
cv2.destroyAllWindows()
