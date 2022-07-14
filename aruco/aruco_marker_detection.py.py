import cv2
import imutils
import numpy as np

ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL}


capture = cv2.VideoCapture(2)
capture.set(3,720)
capture.set(4,480)
cam_mtx = np.array([[503.55091216,   0,         305.63078085],
                    [  0,         503.09411772, 243.44984751],
                    [  0,           0,           1,        ]])
dist_coeff = np.array([[ 1.23526628e-01, -5.69162763e-01,  1.44318047e-04, -1.04153911e-03, 9.04960749e-01]])

while capture.isOpened():

    ret, image = capture.read()
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_1000"])
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
        parameters=arucoParams)
    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            
            # detect marker id only 90
            if markerID == 90:

                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                # marker corner, real size (cm), cam_mtx, dist coeff
                rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorner,14.1, cam_mtx, dist_coeff)
                # rcev : rotation vector
                # tvec : translation vector

                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                cv2.drawFrameAxes(image, cam_mtx, dist_coeff, rvec, tvec, 10)

                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)



                cv2.putText(image,'%.2f, %.2f, %.2f'%(tvec[0,0,0],tvec[0,0,1],tvec[0,0,2]),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

                print("[INFO] ArUco marker ID: {}".format(markerID))
    image = cv2.resize(image,(1280,960))
    cv2.imshow('img',image)
    cv2.waitKey(1)