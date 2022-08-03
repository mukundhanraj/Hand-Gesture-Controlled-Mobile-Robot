import cv2
import recog
import imutils
import numpy as np
import socket

if __name__ == "__main__":
    HOST = "192.168.221.5"  # The server's hostname or IP address
    PORT = 65432  # The port used by the server
    client_socket = socket.socket(
        socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 350, 650

    # initialize num of frames
    num_frames = 0
    calibrated = False
    # keep looping, until interrupted
    while(True):
        # get the current frame
        ret, frame = cap.read()

        if ret:
            # resize the frame
            frame = imutils.resize(frame, width=700)

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

            # get the height and width of the frame
            height, width = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our weighted average model gets calibrated
            if not calibrated:
                if num_frames == 0:
                    print("[STATUS] please wait! calibrating...")
                elif num_frames == 29:
                    calibrated = True
                    print("[STATUS] calibration successfull...")
                recog.run_avg(gray, accumWeight)
                num_frames += 1
            else:
                # segment the hand region
                hand = recog.segment(gray, 15)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    thresholded, segmented, hull = hand
                    thresholded = cv2.morphologyEx(
                        thresholded, cv2.MORPH_OPEN, np.ones((5, 5)))

                    # draw the segmented region and display the frame
                    cv2.drawContours(
                        clone, [hull + (right, top)], -1, (0, 0, 255))

                    angle = recog.getOrientation(segmented, clone, right, top)
                    print(angle)
                    dir = 'NA'
                    if (angle > 75) & (angle < 105):
                        dir = 'straight'
                    elif (angle > 105):
                        dir = 'right'
                    elif (angle < 75):
                        dir = 'left'
                    cv2.putText(clone, dir, (70, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    client_socket.send(bytes(dir, 'utf-8'))

                    # show the thresholded image
                    cv2.imshow("Thesholded", thresholded)

            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q"):
                break

        else:
            break

# free up memory
client_socket.send(b'closing')
client_socket.close()
cap.release()
cv2.destroyAllWindows()
