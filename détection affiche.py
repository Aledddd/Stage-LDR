import cv2
import numpy as np
import pyrealsense2 as rs
import time
from pytessy import pytessy
from PIL import Image
from sklearn.decomposition import PCA
import pickle
import collections
import math



class TargetDetection: #class for detection of specified area

    target_size = (160, 90)
    min_area = 1500

    def get_polygone_points(self, compute_img, nb_polygone_min, nb_polygone_max, order=False, epsilon=0.03): #returns the outlines of the object captured by the camera
    #compute_img : frame captured by camera
    #nb_polygone_min, nb_polygone_max : defines the range of polygones to implement
    #epsilon : coefficient used to reduce the number of points of the outlines
        contours, h = cv2.findContours(cv2.cvtColor(compute_img, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_areas = [0]
        rects = {}
        ret = None
        for c in contours:
            area = cv2.contourArea(c) #stores the number of points which composed the outlines
            if area > self.min_area:
                p = cv2.arcLength(c, True)
                polygone = cv2.approxPolyDP(c, epsilon * p, True) #creates a polygone with 0.03*p number of points of the outlines
                if len(polygone) >= nb_polygone_min and len(polygone) <= nb_polygone_max and cv2.isContourConvex(polygone):#area > np.min(max_areas)
                    if max_areas[0] == 0:
                        max_areas = []
                    rects[int(area)] = np.array(list(map(lambda e: e[0], polygone)), dtype=np.float32)
                    max_areas = np.sort(np.append(max_areas, area)) #adds the value of corresponding area in max_areas tab and sorts it
                    if len(max_areas) > 5:
                        max_areas = max_areas[1:] #deletes the first element of max_areas if it has more than 5 elements
        print(len(rects))
        if len(rects) > 0:
            ret = []
            if order:
                max_areas = sorted(max_areas)
            for max_area in max_areas:
                ret.append(rects[int(max_area)])
                #cv2.drawContours(compute_img, [np.array([[e] for e in rects[int(max_area)]], dtype=int)], 0, (255, 0, 0), 2)
        #cv2.imshow("get_polygone_points", compute_img)
        return ret

    def compute_img_for_detection(self, compute_img, blur, erode, dilate, mask_low, mask_high):
        #compute_img += 1
        if blur:
            compute_img = cv2.GaussianBlur(compute_img, (blur, blur), 0) #filters the computed image by convolving each point with a Gaussian Kernal
        mask = cv2.inRange(compute_img, mask_low, mask_high) #defines the range of colors to better detect the corresponding object
        compute_img = cv2.bitwise_and(compute_img, compute_img, mask=mask) #compars the computed image and itself filtered
        compute_img = cv2.cvtColor(compute_img, cv2.COLOR_HSV2BGR) #converts the computed image from HSV (seen by camera) to BGR (processed by algo)
        compute_img = cv2.erode(compute_img, None, iterations=erode) #erodes away the boundaries of foreground object to diminish the features of an image.
        compute_img = cv2.dilate(compute_img, None, iterations=dilate) #increases the white region in the image to accentuate features
        return compute_img

    def getCenter(self, rect): #rect : order of the points corresponding to the outlines of the object
        return (((rect[1] - rect[3]) / 2) + rect[3]).astype(int)

    def getZoomedRect(self, rect):
        center = self.getCenter(rect)
        return np.array(((rect-center)*self.quick_scan_frame_coeff)+center, dtype=int)

    def cutFromContour(self, img, rect, size):
        target = np.float32([[0, 0], [0, size[1]], [size[0], size[1]], [size[0], 0]]) #constructs the set of destination points
        transform_matrix = cv2.getPerspectiveTransform(np.float32(rect), target)
        return cv2.warpPerspective(img, transform_matrix, size)

    def getLevelContour(self, rect):
        maxi = np.max(rect, 0)
        mini = np.min(rect, 0)
        self.target_size = (int(maxi[0]-mini[0]), int(maxi[1]-mini[1]))
        return np.array([[mini[0], mini[1]], [mini[0], maxi[1]], [maxi[0], maxi[1]], [maxi[0], mini[1]]])

    def extract_target(self, img, rect):
        if cv2.norm(rect[0] - rect[3]) < cv2.norm(rect[0] - rect[1]):
            rect = np.roll(rect, 2)
        return self.cutFromContour(img, rect, (160, 90))



class WindowDetection(TargetDetection):

    quick_scan_frame_coeff = 2.5
    counter_fail_detection = 0
    counter_fail_detection_limit = 3
    last_seen_area = None
    min_area = 15000

    def get_target(self, img):
        img = cv2.bitwise_not(img)
        img -= 1
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        cv2.imshow("window_", cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
        computed_img = self.compute_img_for_detection(img, 0, 2, 27, np.array([0, 0, 240]), np.array([180, 15, 255]))
        if self.last_seen_area is not None:
            computed_img = self.cutFromContour(computed_img, self.last_seen_area, tuple(np.array(self.quick_scan_frame_coeff * np.array(self.target_size), dtype=int)))
        rect_list = self.get_polygone_points(computed_img, 4, 4, True, 0.05)
        cv2.imshow("window", computed_img)
        if rect_list is not None:
            for rect in rect_list:
                if self.check_square(rect) > 40: continue

                if self.last_seen_area is not None:
                    rect = rect + self.last_seen_area[0]
                target = self.extract_target(img, rect)

                self.last_seen_area = self.getZoomedRect(self.getLevelContour(rect))
                self.counter_fail_detection = 0
                return target, self.getCenter(rect)
        if self.counter_fail_detection > self.counter_fail_detection_limit:
            self.last_seen_area = None
        self.counter_fail_detection += 1
        return None, (0, 0)

    def check_square(self, rect):
        side_len = np.zeros(4)
        for i in range(4):
            side_len[i] = math.sqrt(pow(rect[i-1][0]-rect[i][0], 2) + pow(rect[i-1][1]-rect[i][1], 2))
        return np.std(side_len)


# Lancement du live
pipeline = rs.pipeline()

    # Create a config object
config = rs.config()
    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    #s.config.enable_device_from_file(config, args.input)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

windowDetection = WindowDetection()

while True:
    # Get frameset of depth
    frames = pipeline.wait_for_frames()

    # Get depth frame
    color_frame = frames.get_color_frame()

    # Validate that both frames are valid
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())

    t4, c4 = windowDetection.get_target(color_image)
    if c4 is not None:
        cv2.putText(color_image, "x-WINDOW", (c4[0] - 9, c4[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
    else:
        continue
    cv2.imshow('color',color_image)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        pipeline.stop()
        cv2.destroyAllWindows()
        break