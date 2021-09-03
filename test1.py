# Python program to write
# text on video


import cv2

cap = cv2.VideoCapture(0)
i=0
while (True):

    # Capture frames in the video
    ret, frame = cap.read()

    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for
    # inserting text on video
    i = i+1
    if i % 30 == 0:
        print(i)
        cv2.putText(frame,
                    '{:d}'.format(i),
                    (50, 50),
                    font, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)
        cv2.imshow('video', frame)
    cv2.putText(frame,
                'TEST',
                (100, 100),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)
    # Display the resulting frame
    #cv2.imshow('video', frame)

    # creating 'q' as the quit
    # button for the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()
