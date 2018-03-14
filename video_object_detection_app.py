import os,sys
import cv2
import time,datetime
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import pandas as pd

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'ssd_mobilenet_man_woman'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
#PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'man_woman_labels.pbtxt')


#NUM_CLASSES = 2
NUM_CLASSES = 90
object_list = []

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#print ("label categories : ",category_index[1]['name'])


def detect_objects(image_np, sess, detection_graph, write_q):
	#get_class_label_dictionary(class_dict, PATH_TO_LABELS)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    curr_ts = datetime.datetime.now()
    #print ("scores shape: ", scores[0].shape,scores.shape,classes.shape)
    for i in range(100):
        if scores[0][i] > 0.6:
           myrow = str(curr_ts)+","+str(scores[0][i])+","+str(int(classes[0][i]))+","+category_index[classes[0][i]]['name']
           print(myrow)
           write_q.put(myrow)
		#print ("number of objects : ",str(len(object_list)))
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np

def write_summary(objdet_sum_file, write_q,end_of_work):
    with open(objdet_sum_file, 'w') as dest_file:
        while True:
            line = write_q.get()
            if line == end_of_work:
                return
            dest_file.write(line+"\n")


def worker(input_q, output_q, write_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph, write_q))
        #print ("obj sum : ",writer_q.get())
        #write_summary('C:\\arun\\objdet_sum.csv',writer_q)
    fps.stop()
    sess.close()


if __name__ == '__main__':
    print ("nbr of args : ", len(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=str,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=5, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=100, help='Size of the queue.')
    args = parser.parse_args()    

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    write_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q, write_q))

    #video_capture = WebcamVideoStream(src='c:\\arun\\object_detector_app\\big_buck_bunny.mp4',
    #                                  width=args.width,
    #                                  height=args.height).start()
    #cap = cv2.VideoCapture('C:\\arun\\object_detector_app\\big_buck_bunny.mp4')
    videofile = 'C:\\arun\\videos\\people_walking.mp4'
    print("Start reading the frames from Video")  
    cap = cv2.VideoCapture(videofile)
    fps = FPS().start()
    end_of_work = "END_OF_WORK"
    writer_process = multiprocessing.Process(target = write_summary, args=("C:\\arun\\object_detection_summary.txt", write_q, end_of_work))
    writer_process.start()
	
    while (cap.isOpened()):  # fps._numFrames < 120
        ret,frame = cap.read()
        if ret == True:
            print("Frame is read successfully")
            input_q.put(frame)
            t = time.time()

            output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', output_rgb)
            fps.update()

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print ("the while loop is breaking")
                break
        else:
            print (" End of the Video")
            print ("total number of objects in video : ",str(len(object_list)))
            break     		

    print(" calling fps stop")
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    print ("total number of objects detected : ",str(len(object_list)))

    pool.terminate()
    cap.release()
    cv2.destroyAllWindows()
    write_q.put(end_of_work)
    writer_process.join()