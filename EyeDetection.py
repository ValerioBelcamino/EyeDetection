import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


def handle_close(event, cap):
    cap.release()


def bgr_to_rgb(image):
    """
    Convert a BGR image into RBG
    :param image: the BGR image
    :return: the same image but in RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    closed = False
    right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

    cap = cv2.VideoCapture(0)
    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))
    img = None
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rep = 0

        right_eye = right_eye_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in right_eye:
            if rep == 0:

                rep=rep+1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_grayR = gray[y:y + h, x:x + w]
                # roi_colorR = frame[y:y + h, x:x + w]

        rep = 0
        left_eye = left_eye_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in left_eye:
            if rep == 0:
                #print("ciao")
                rep=rep+1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_grayL = gray[y:y + h, x:x + w]
                roi_colorL = frame[y:y + h, x:x + w]
                rowsL, colsL, _ = roi_colorL.shape
        #print(roi_grayL.shape[0] // 2)
        newimg = roi_grayL[roi_grayL.shape[0] // 2 :, :]
        newimg = cv2.equalizeHist(newimg)
        # newimg= newimg//2*5
        # newimg = cv2.GaussianBlur(newimg, (2, 2), 0)

        newimg = cv2.bilateralFilter(newimg,8,150,150)
        newimg = cv2.medianBlur(newimg, 3)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 6))
        newimg = cv2.morphologyEx(newimg, cv2.MORPH_CLOSE, kernel, iterations=1)
        avg = np.average(newimg)
        newimg = cv2.threshold(newimg, 35, 255, cv2.THRESH_BINARY_INV)[1]
        newimg = cv2.erode(newimg, kernel2, iterations=3)

        #newimg = cv2.morphologyEx(newimg, cv2.MORPH_CLOSE, kernel, iterations=1)

        #newimg = cv2.dilate(newimg, kernel2, iterations=2)
        #print(newimg.shape)

        contours, hierarchy = cv2.findContours(newimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        #print(contours)
        imgcont = cv2.drawContours(newimg, contours, -1, (1, 0, 0), 3)
        (x, y, w, h) = cv2.boundingRect(contours[0])
        #cv2.rectangle(newimg, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #print(cv2.contourArea(contours[0]))
        if cv2.contourArea(contours[0]) < 100 and not closed:
            closed = True
            timer = time.time()
            print(closed, timer)
        elif closed:
            if time.time() - timer > 3:
                print("occhio chiusoooooo")
                closed = False

            # print("occhio chiuso")
        cv2.imshow("roi", roi_colorL)


        if img is None:

            # img=plt.imshow( imgcont )
            img = plt.imshow(newimg, cmap='gray', vmin=0, vmax=255)
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            # show the plot!
            plt.show()
        else:
            # set the current frame as the data to show
            img.set_data(newimg)
            # update the figure associated to the shown plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(1/30)  # pause: 30 frames per second

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
