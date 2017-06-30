import cv2
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
#from moviepy.editor import *
import numpy as np
import time
#cap = cv2.VideoCapture('road.avi')
#print(cap)
#while(cap.isOpened()):
#    print("hello")
#    ret, frame = cap.read()
#   print(frame.shape)
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#    cv2.imshow('frame',gray)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
cv2.destroyAllWindows()

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=5.)


# Convolutional network building
network = input_data(shape=[None, 100, 100, 3],data_preprocessing=img_prep,data_augmentation=img_aug)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)



model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('model/newmodels.tfl')
image = cv2.imread('t3.jpg')
stepSiz=5
(winW, winH)=(100,100)

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0],:])
			
all_rec = []
for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
#     print(x,"\t",y,"\t",window.shape)
#     img = cv2.cvtColor(window,cv2.COLOR_GRAY2BGR)
#     cv2.imshow(img)
    
#     print(window.dtype)
    
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    img=np.reshape(window,(1,100,100,3))
#     print(img.shape)
#     print(img.dtype)
    y_predict = model.predict(np.array(img,dtype=np.float32))
    label = np.argmax(y_predict)
#     print(label)
    if label!= 3:
        all_rec.append([x,y,x+winW,y+winH])
    
#         cv2.rectangle(image, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
#         cv2.imshow("Window", image)
#         cv2.waitKey(1)
#         time.sleep(0.025)
print(np.array(all_rec).shape)

gr_rect= cv2.groupRectangles(all_rec,1,0.4)
print(np.array(all_rec).shape)
print(np.array(gr_rect[0]).shape)

while(True):
    for i in range(np.array(gr_rect[0]).shape[0]):
        cv2.imshow("Window", image)
        cv2.rectangle(image, (gr_rect[0][i][0], gr_rect[0][i][1]), (gr_rect[0][i][2], gr_rect[0][i][3]), (0, 255, 0), 2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

gr_rect= cv2.groupRectangles(all_rec,1,0.1)
print(np.array(all_rec).shape)
print(np.array(gr_rect[0]).shape)
