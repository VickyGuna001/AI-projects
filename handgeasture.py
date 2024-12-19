import cv2
import mediapipe as mp
# Initialize Mediapipe Hands model
np_hands=mp.solutions.hands # pre trained model for hand detection & landmark identification
np_drawing=mp.solutions.drawing_utils# provides utilities to draw landmarks(eg:hand points) and connections on line detected hand
cap=cv2.VideoCapture(0)# intillize the camera
with np_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.5) as hands:# confidece is used to understanding percentage
    while True:
        ref,frame=cap.read()# ref give values and frame give pic 
        if not ref:
            break
        frame=cv2.flip(frame,1)# fip the pic like mirror image & the frame horizontal for better view
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)# convert the color bgr to rgb formate because mediapipe want rgb formate
        # process the frame and detect the hand
        result=hands.process(rgb_frame)
        # if hand detected in the frame
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # draw the landmark on the hand
                np_drawing.draw_landmarks(frame,hand_landmarks,np_hands.HAND_CONNECTIONS)
        # display the frame with the hand landmark        
        cv2.imshow("HAND GEASTURE RECOGNITION",frame)
        key=cv2.waitKey(1) # frame wait for 1 sec 
        if key==27:#if press esc it will break the loop
            break
# release the wibcom and close the window        
cap.release()                
cv2.destroyAllWindows()