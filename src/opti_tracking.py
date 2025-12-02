import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Create old frame
_ , frame = cap.read()
old_grey = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

# Lucas Kanade Params
lk_params = dict(winSize = (10 , 10) , maxLevel = 4 , 
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT , 10 , 0.03))

# Mouse functions

def select_point(event , x , y , flags , params):
    global point , point_selected , old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x , y)
        point_selected = True
        old_points = np.array([[x , y]] , dtype = np.float32)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame" , select_point)

point_selected = False
point = ()
old_points = np.array([[]])

while True:
    _ , frame = cap.read()
    grey_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    if point_selected is True:
        cv2.circle(frame , point , 5 , (0 , 0 , 255) , 2)

        new_points , status , error = cv2.calcOpticalFlowPyrLK(old_grey , grey_frame , old_points, None , **lk_params)
        old_grey = grey_frame.copy()
        old_points = new_points

        x, y = new_points.ravel()
        cv2.circle(frame , (int(x), int(y)) , 5 , (0 , 255 , 0) , 1)     

    first_level = cv2.pyrDown(frame)

    cv2.imshow("Frame" , frame )

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()