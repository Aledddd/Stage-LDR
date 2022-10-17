import numpy as np
import cv2.cv2

# Matrice de taille 5*5, elle permet de faire de la convolution sur notre image pour en modifier ses caractéristiques
kernel = np.ones((5, 5), np.uint8) / 25

# Fonction de détection d'un cercle en détectant les contours
def getContours(img):
    contours, hierarchy = cv2.cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            #cv2.drawContours(img, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            objectCorners = len(approx)
            # On considère la forme comme un cercle à partir de 5 coins, et on retourne ses informations
            #print(objectCorners)
            if objectCorners > 10:
                    return "OK"
    # Retour par défaut si aucun cercle n'a été trouvé
    return "NOT OK"


#cap = cv2.VideoCapture('http://user:pass@192.168.43.190:8000/stream.mjpg')
cap = cv2.VideoCapture(0)

# CIBLE SUR L'IMAGE PARAMETRABLE
targetPointX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
targetPointY = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)


def empty(a):
    pass


#Trackbar avec slider
cv2.namedWindow("Trackbar")
cv2.resizeWindow("Trackbar",(300,900))
# Blue
cv2.createTrackbar("Hue Min Blue","Trackbar",90,180,empty)
cv2.createTrackbar("Hue Max Blue","Trackbar",128,180,empty)
cv2.createTrackbar("Sat Min Blue","Trackbar",50,255,empty)
cv2.createTrackbar("Sat Max Blue","Trackbar",255,255,empty)
cv2.createTrackbar("Val Min Blue","Trackbar",70,255,empty)
cv2.createTrackbar("Val Max Blue","Trackbar",255,255,empty)
# Red
cv2.createTrackbar("Hue Min Red","Trackbar",159,180,empty)
cv2.createTrackbar("Hue Max Red","Trackbar",180,180,empty)
cv2.createTrackbar("Sat Min Red","Trackbar",50,255,empty)
cv2.createTrackbar("Sat Max Red","Trackbar",255,255,empty)
cv2.createTrackbar("Val Min Red","Trackbar",70,255,empty)
cv2.createTrackbar("Val Max Red","Trackbar",255,255,empty)
# Yellow
cv2.createTrackbar("Hue Min Yellow","Trackbar",25,180,empty)
cv2.createTrackbar("Hue Max Yellow","Trackbar",35,180,empty)
cv2.createTrackbar("Sat Min Yellow","Trackbar",50,255,empty)
cv2.createTrackbar("Sat Max Yellow","Trackbar",255,255,empty)
cv2.createTrackbar("Val Min Yellow","Trackbar",70,255,empty)
cv2.createTrackbar("Val Max Yellow","Trackbar",255,255,empty)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()
    output_frame = image.copy()

    #Application des Trackbars à l'image
    #Blue
    h_min_b = cv2.getTrackbarPos("Hue Min Blue","Trackbar")
    h_max_b = cv2.getTrackbarPos("Hue Max Blue","Trackbar")
    s_min_b = cv2.getTrackbarPos("Sat Min Blue","Trackbar")
    s_max_b = cv2.getTrackbarPos("Sat Max Blue","Trackbar")
    v_min_b = cv2.getTrackbarPos("Val Min Blue","Trackbar")
    v_max_b = cv2.getTrackbarPos("Val Max Blue","Trackbar")
    # Red
    h_min_r = cv2.getTrackbarPos("Hue Min Red","Trackbar")
    h_max_r = cv2.getTrackbarPos("Hue Max Red","Trackbar")
    s_min_r = cv2.getTrackbarPos("Sat Min Red","Trackbar")
    s_max_r = cv2.getTrackbarPos("Sat Max Red","Trackbar")
    v_min_r = cv2.getTrackbarPos("Val Min Red","Trackbar")
    v_max_r = cv2.getTrackbarPos("Val Max Red","Trackbar")
    # Yellow
    h_min_y = cv2.getTrackbarPos("Hue Min Yellow","Trackbar")
    h_max_y = cv2.getTrackbarPos("Hue Max Yellow","Trackbar")
    s_min_y = cv2.getTrackbarPos("Sat Min Yellow","Trackbar")
    s_max_y = cv2.getTrackbarPos("Sat Max Yellow","Trackbar")
    v_min_y = cv2.getTrackbarPos("Val Min Yellow","Trackbar")
    v_max_y = cv2.getTrackbarPos("Val Max Yellow","Trackbar")
    #print(h_min,h_max,s_min,s_max,v_min,v_max)



    kernel = np.ones((5, 5), np.uint8) / 25

    # FPS de la vidéo
    cv2.putText(output_frame, "FPS : " + str(cap.get(cv2.CAP_PROP_FPS)), (image.shape[1] - 100, 15),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)

    #Cette fonction permet de réduire le bruit sur l'image
    image_bgr = cv2.cv2.bilateralFilter(image,5,200,200)
    #L'érosion et la dilation permettent de volontairement dégrader l'image afin d'en faire ressortir des éléments
    image_bgr = cv2.erode(image_bgr, kernel, iterations=3)
    image_bgr = cv2.dilate(image_bgr,kernel,iterations=5)
    cv2.imshow('erode',image_bgr)
    #On convertit dans le format HSV pour pouvoir toucher à la teinte, la saturation et la luminosité de la couleur
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    cv2.imshow('raw',image_hsv)
    # Possible yellow threshold: [20, 110, 170][20, 110, 170]
    # Possible blue threshold: [20, 115, 70][255, 145, 120]
    #Il s'agit d'un masque, cela permet de faire ressortir uniquement les couleurs souhaités en fonction d'un interval donné
    image_hsv_blue = cv2.inRange(image_hsv, np.array([ h_min_b, s_min_b, v_min_b]), np.array([h_max_b, s_max_b, v_max_b]))
    image_hsv_yellow = cv2.inRange(image_hsv, np.array([h_min_y, s_min_y, v_min_y]), np.array([h_max_y, s_max_y, v_max_y]))
    image_hsv_red = cv2.inRange(image_hsv, np.array([h_min_r, s_min_r, v_min_r]), np.array([h_max_r, s_max_r, v_max_r]))
    # np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]) Test
    # On applique du flou sur l'image pour réduire davantage le bruit
    image_hsv_red = cv2.GaussianBlur(image_hsv_red, (9, 9), 2, 2)
    image_hsv_yellow = cv2.GaussianBlur(image_hsv_yellow, (9, 9), 2, 2)
    image_hsv_blue = cv2.GaussianBlur(image_hsv_blue, (9, 9), 2, 2)

    #Cette fonction compare la différence entre deux éléments
    #Dans notre cas, nous appliquons un masque sur la comparaison pour ne faire ressortir uniquement la couleur associé à ce masque
    image_hsv_red_visu = cv2.bitwise_and(image_hsv,image_hsv,mask=image_hsv_red)
    image_hsv_yellow_visu = cv2.bitwise_and(image_hsv,image_hsv,mask=image_hsv_yellow)
    image_hsv_blue_visu = cv2.bitwise_and(image_hsv,image_hsv,mask=image_hsv_blue)

    image_hsv_red = cv2.Canny(image_hsv_red_visu,75,75)
    image_hsv_yellow = cv2.Canny(image_hsv_yellow_visu, 75, 75)
    image_hsv_blue = cv2.Canny(image_hsv_blue_visu, 75, 75)
    #On visualise les résultats
    cv2.imshow('blue',image_hsv_blue)
    cv2.imshow('red', image_hsv_red)
    cv2.imshow('yellow', image_hsv_yellow)

    #On applique la fonction HoughCircles pour détecter les cercles dans l'image, la fonction renvoit les coordonnées du centre du cercle ainsi que sa taille en pixel
    #La fonction fonctionne sur des images en grayscale ce qui est le cas de mes masques qui font ressortir en blanc les éléments que l'on veut détecter
    circles_red = cv2.HoughCircles(image_hsv_red, cv2.HOUGH_GRADIENT,image_hsv.shape[0]/64, image_hsv.shape[0]/8, param1=100, param2=30,minRadius=0, maxRadius=0)
    circles_yellow = cv2.HoughCircles(image_hsv_yellow,cv2.HOUGH_GRADIENT, image_hsv.shape[0]/64,  image_hsv.shape[0]/8, param1=100, param2=30,minRadius=0, maxRadius=0)
    circles_blue = cv2.HoughCircles(image_hsv_blue, cv2.HOUGH_GRADIENT, image_hsv.shape[0]/64,  image_hsv.shape[0]/8, param1=100, param2=30,minRadius=0, maxRadius=0)

    #On désigne le milieu de l'image comme étant la cible à atteindre
    cv2.circle(output_frame,(targetPointX,targetPointY),1,(0,0,0),3)
    # Message pour les instructions envoyées au drone
    messageX, messageY = "none", "none"


    if circles_red is not None:
        if getContours(image_hsv_red) =='OK':
            circles_red = np.round(circles_red[0, :]).astype("int")
            cv2.circle(output_frame, center=(circles_red[0, 0], circles_red[0, 1]), radius=circles_red[0, 2], color=(0, 0, 255), thickness=2)
            cv2.circle(output_frame,center=(circles_red[0, 0], circles_red[0, 1]),radius=2,color=(0, 0, 255), thickness=2)

            if circles_red[0,0] < targetPointX - 10:
                messageX = "GAUCHE"
            elif targetPointX + 10 < circles_red[0,0]:
                messageX = "DROITE"
            else:
                messageX = "OK"

            if circles_red[0,1] < targetPointY - 10:
                messageY = "DEVANT"
            elif targetPointY + 10 < circles_red[0,1]:
                messageY = "DERRIERE"
            else:
                messageY = "OK"
            # Si le cercle est sur la cible
            if messageX == "OK" and messageY == "OK":
                cv2.putText(output_frame,"Cible atteinte",(targetPointX,targetPointY),cv2.FONT_HERSHEY_COMPLEX, 0.5,(255,255,255),1)
                # Affichage de la position du cercle dans l'image
                cv2.putText(output_frame, "Circle Position : (" + str(circles_red[0,0]) + ";" +  str(circles_red[0,1]) + ")",
                            (output_frame.shape[1] - 250, output_frame.shape[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),1)
                print("INSTRUCTIONS\tDESCENTE")
            else:
                # Affichage de la position du cercle dans l'image
                cv2.putText(output_frame, "Circle Position : (" + str(circles_red[0,0]) + ";" + str(circles_red[0,1]) + ")",
                            (output_frame.shape[1] - 250, output_frame.shape[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0,0,255), 1)
                cv2.circle(output_frame, (targetPointX, targetPointY), 10, (0, 0, 0), 2)
                # Envoi des instructions dans la console
                print("INSTRUCTIONS\tX : " + messageX + " Y : " + messageY)

                # Affichage de la position de la cible dans l'image
                cv2.putText(output_frame, "Target Position : (" + str(targetPointX) + ";" + str(targetPointY) + ")",
                            (1, output_frame.shape[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    if circles_yellow is not None:
        if getContours(image_hsv_yellow) == 'OK':
            circles_yellow = np.round(circles_yellow[0, :]).astype("int")
            cv2.circle(output_frame, center=(circles_yellow[0, 0], circles_yellow[0, 1]), radius=circles_yellow[0, 2], color=(0, 255, 255), thickness=2)
            cv2.circle(output_frame,center=(circles_yellow[0, 0], circles_yellow[0, 1]),radius=2,color=(0, 255, 255), thickness=2)

            if circles_yellow[0, 0] < targetPointX - 10:
                messageX = "GAUCHE"
            elif targetPointX + 10 < circles_yellow[0, 0]:
                messageX = "DROITE"
            else:
                messageX = "OK"

            if circles_yellow[0, 1] < targetPointY - 10:
                messageY = "DEVANT"
            elif targetPointY + 10 < circles_yellow[0, 1]:
                messageY = "DERRIERE"
            else:
                messageY = "OK"
                # Si le cercle est sur la cible
            if messageX == "OK" and messageY == "OK":
                cv2.putText(output_frame, "Cible atteinte", (targetPointX, targetPointY), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (255, 255, 255), 1)
                # Affichage de la position du cercle dans l'image
                cv2.putText(output_frame,
                            "Circle Position : (" + str(circles_yellow[0, 0]) + ";" + str(circles_yellow[0, 1]) + ")",
                            (output_frame.shape[1] - 250, output_frame.shape[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (255, 255, 255), 1)
                print("INSTRUCTIONS\tDESCENTE")
            else:
                # Affichage de la position du cercle dans l'image
                cv2.putText(output_frame,
                            "Circle Position : (" + str(circles_yellow[0, 0]) + ";" + str(circles_yellow[0, 1]) + ")",
                            (output_frame.shape[1] - 250, output_frame.shape[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 255), 1)
                cv2.circle(output_frame, (targetPointX, targetPointY), 10, (0, 0, 0), 2)
                # Envoi des instructions dans la console
                print("INSTRUCTIONS\tX : " + messageX + " Y : " + messageY)

                # Affichage de la position de la cible dans l'image
                cv2.putText(output_frame, "Target Position : (" + str(targetPointX) + ";" + str(targetPointY) + ")",
                            (1, output_frame.shape[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    if circles_blue is not None:
        if getContours(image_hsv_blue) == 'OK':
            circles_blue = np.round(circles_blue[0, :]).astype("int")
            cv2.circle(output_frame, center=(circles_blue[0, 0], circles_blue[0, 1]), radius=circles_blue[0, 2], color=(255, 0, 0), thickness=2)
            cv2.circle(output_frame, center=(circles_blue[0, 0], circles_blue[0, 1]), radius=2, color=(255, 0, 0),thickness=2)

            if circles_blue[0, 0] < targetPointX - 10:
                messageX = "GAUCHE"
            elif targetPointX + 10 < circles_blue[0, 0]:
                messageX = "DROITE"
            else:
                messageX = "OK"

            if circles_blue[0, 1] < targetPointY - 10:
                messageY = "DEVANT"
            elif targetPointY + 10 < circles_blue[0, 1]:
                messageY = "DERRIERE"
            else:
                messageY = "OK"
                # Si le cercle est sur la cible
            if messageX == "OK" and messageY == "OK":
                cv2.putText(output_frame, "Cible atteinte", (targetPointX, targetPointY), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (255, 255, 255), 1)
                # Affichage de la position du cercle dans l'image
                cv2.putText(output_frame,
                            "Circle Position : (" + str(circles_blue[0, 0]) + ";" + str(circles_blue[0, 1]) + ")",
                            (output_frame.shape[1] - 250, output_frame.shape[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (255, 255, 255), 1)
                print("INSTRUCTIONS\tDESCENTE")
            else:
                # Affichage de la position du cercle dans l'image
                cv2.putText(output_frame,
                            "Circle Position : (" + str(circles_blue[0, 0]) + ";" + str(circles_blue[0, 1]) + ")",
                            (output_frame.shape[1] - 250, output_frame.shape[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 255), 1)
                cv2.circle(output_frame, (targetPointX, targetPointY), 10, (0, 0, 0), 2)
                # Envoi des instructions dans la console
                print("INSTRUCTIONS\tX : " + messageX + " Y : " + messageY)

                # Affichage de la position de la cible dans l'image
                cv2.putText(output_frame, "Target Position : (" + str(targetPointX) + ";" + str(targetPointY) + ")",
                            (1, output_frame.shape[0] - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)


    cv2.imshow('frame', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#On stoppe et ferme le flux vidéo
cap.release()
cv2.destroyAllWindows()