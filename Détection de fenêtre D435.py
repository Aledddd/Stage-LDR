import cv2
import numpy as np
import pyrealsense2 as rs


#Préparation du flux vidéo Realsense D435
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
pipeline.start(config)
colorizer = rs.colorizer()

targetPointX = 640 // 2
targetPointY = 480 // 2

while True:
    #Récupération des données vidéos
    frames = pipeline.wait_for_frames()
    color_frames = frames.get_color_frame()
    #Récupération des données vidéos en format lisible par OpenCV
    color_image = np.asanyarray(color_frames.get_data())
    image = color_image.copy()
    result = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    result = cv2.inRange(result,np.array([0,0,0]),np.array([180, 255, 30]))
    '''np.array([0,0,0]),np.array([180, 255, 30])'''
    canny = cv2.erode(result,(7,7))
    canny = cv2.dilate(canny,(7,7))
    maxArea = 5000
    contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.circle(image, (targetPointX,targetPointY), 10, (0, 0, 0), 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area > maxArea:
            perimeter = cv2.arcLength(cnt,True)

            approx = cv2.approxPolyDP(cnt,0.01*perimeter,True)
            objectCorners = len(approx)

            x, y, width, height = cv2.boundingRect(approx)

            if objectCorners == 4:
                aspRatio = width / float(height)
                aspRatio = np.round(aspRatio,3)
                if aspRatio > 0.85 and aspRatio < 1.15:
                    print(aspRatio)
                    cv2.circle(image, (int(x) + int(width) // 2, int(y) + int(height) // 2), 10, (0, 0, 0), 2)
                    objectType = "Square"
                    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 0), 5)
                    #cv2.putText(image, objectType, (x + width // 2, y + height + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                     #       (0, 255, 0), 2)
                    cv2.putText(image, str(str(width)+" px,"+str(height)+" px"),(x + width // 2, y + height + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 2)
                    if x == "none":
                        centerWinX = "none"
                        centerWinY = "none"
                    else:
                        centerWinX = int(x) + int(width) // 2
                        centerWinY = int(y) + int(height) // 2

                    cv2.putText(image, "Window position : ( " + str(centerWinX) + " ; " + str(centerWinY) + " )",
                                (10, image.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    messageX = ""
                    messageY = ""
                    targetPointXMin = image.shape[1] // 2 - 10
                    targetPointXMax = image.shape[1] // 2 + 10
                    targetPointYMin = image.shape[0] // 2 - 10
                    targetPointYMax = image.shape[0] // 2 + 10

                    if centerWinX == "none":
                        messageX = "none"
                    else:
                        if int(centerWinX) < targetPointXMin:
                            messageX = "DROITE"
                        elif int(centerWinX) > targetPointXMax:
                            messageX = "GAUCHE"
                        else:
                            messageX = "OK"

                    if centerWinY == "none":
                        messageY = "none"
                    else:
                        if int(centerWinY) < targetPointYMin:
                            messageY = "BAS"
                        elif int(centerWinY) > targetPointYMax:
                            messageY = "HAUT"
                        else:
                            messageY = "OK"

                    print("INSTRUCTIONS\tX : " + messageX + "\tY : " + messageY)
                    if messageX == "OK" and messageY == "OK":
                        cv2.circle(image, (targetPointX, targetPointY), 10, (255, 255, 255), 2)
                        print("AVANCER")
                '''else:
                    objectType = "No Square"
                    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 3)
                    cv2.putText(image, objectType, (x + width // 2, y + height + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0,0,255), 2)
            else:
                objectType = "No square/rectangle"
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 3)
                cv2.putText(image, objectType, (x + width // 2, y + height + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 2)'''

    #cnt = contours[4]
    #cv2.drawContours(color_image, [cnt], 0, (0, 255, 0), 3)
    #depth_image = cv2.bitwise_and(canny,depth_image)
    cv2.imshow('grey',color_image)
    cv2.imshow('color image', image)
    #cv2.imshow('gdrt', gradient)
    cv2.imshow('canny', canny)
    #cv2.imshow('colored depth', depth_image)
    cv2.imshow('result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.stop()
        cv2.destroyAllWindows()
        break