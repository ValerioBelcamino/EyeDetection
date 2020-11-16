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

def transformsR(image):

    """
    Apply several transformations to the original image:
        blur, equalization, tresholding, morphological operators
    :param image: the B/W image
    :return: thresholded image
    """

    newimg = image[image.shape[0] // 2:, :]
    newimg = cv2.equalizeHist(newimg)
    newimg = cv2.bilateralFilter(newimg, 8, 150, 150)
    newimg = cv2.medianBlur(newimg, 3)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))
    newimg = cv2.morphologyEx(newimg, cv2.MORPH_CLOSE, kernel, iterations=1)
    avg = np.average(newimg)
    newimg = cv2.threshold(newimg, 35, 255, cv2.THRESH_BINARY_INV)[1]
    newimg = cv2.erode(newimg, kernel2, iterations=3)

    newimg = cv2.resize(newimg, (80, 40), interpolation=cv2.INTER_NEAREST)
    return newimg


def transformsL(image):

    """
    Apply several transformations to the original image:
        blur, equalization, tresholding, morphological operators
    :param image: the B/W image
    :return: thresholded image
    """

    newimg = image[image.shape[0] // 2:, :]
    newimg = cv2.equalizeHist(newimg)
    newimg = cv2.bilateralFilter(newimg, 8, 150, 150)
    newimg = cv2.medianBlur(newimg, 3)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
    newimg = cv2.morphologyEx(newimg, cv2.MORPH_CLOSE, kernel, iterations=1)
    avg = np.average(newimg)
    newimg = cv2.threshold(newimg, 35, 255, cv2.THRESH_BINARY_INV)[1]
    newimg = cv2.erode(newimg, kernel2, iterations=3)

    newimg = cv2.resize(newimg, (80,40) ,interpolation = cv2.INTER_NEAREST)
    return newimg



def checkContours(img):

    """
    checks if there's a contours big enough to be our iris
    :param image: the threshold image
    :return: boolean
    """

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    imgcont = cv2.drawContours(img, contours, -1, (1, 0, 0), 3)
    #(x, y, w, h) = cv2.boundingRect(contours[0])
    # cv2.rectangle(newimg, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #print(cv2.contourArea(contours[0]) )
    try:
        bool = cv2.contourArea(contours[0]) < 100
    except IndexError:
        print("errore index")
        return False;
    return bool


def cutROI(img, classifier):

    """
    returns the ROI of the eye from the original B/W image
    :param image: the threshold image
    :return: boolean
    """

    rep = 0

    roi_gray = None

    right_eye = classifier.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in right_eye:
        if rep == 0:
            rep = rep + 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = img[y:y + h, x:x + w]

    return roi_gray


def printTextOnImage(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX

    textsize = cv2.getTextSize(text, font, 4, 5)[0]
    textX = (img.shape[1] - textsize[0]) // 2
    textY = (img.shape[0] + textsize[1]) // 2
    cv2.putText(img, 'Eye not found', (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5, cv2.LINE_AA)


def main():

    # CASCADE CLASSIFIER CREATION

    closed = False
    right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
    # GET CAM HANDLE AND PREPARE EXIT FUNCTION

    cap = cv2.VideoCapture(0)
    plt.rcParams['toolbar'] = 'None'

    plt.ion()
    fig = plt.figure()
    # Creare your figure and axes
    fig, ax = plt.subplots(1)

    # Set whitespace to 0
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))
    img = None
    found = False

    # MAIN CYCLE

    while cap.isOpened():

        # READ FIRST FRAME AND GET B/W COPY

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        colorImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        roi_face = cutROI(gray, face_cascade)
        if roi_face is not None:
            roi_faceR = roi_face[:, : roi_face.shape[0] // 2]
            roi_faceL = roi_face[:, roi_face.shape[0] // 2 :]

            roi_grayR = cutROI(roi_faceR, right_eye_cascade)

            roi_grayL = cutROI(roi_faceL, left_eye_cascade)



        if roi_grayL is not None and roi_grayR is not None:

            newimgL = transformsL(roi_grayL)

            newimgR = transformsR(roi_grayR)


            if checkContours(newimgL) and checkContours(newimgR) and not closed:
                closed = True
                timer = time.time()
                print(closed, timer)
            elif closed:
                if time.time() - timer > 3:
                    print("occhio chiusoooooo")
                    closed = False
        else:
            print("non trovato")
            printTextOnImage(colorImg, "Eye not found!")

        if img is None:

            ax = plt.gca()
            # get axis extent
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

            img = plt.imshow(newimgR, aspect='equal')
            plt.axis("off")  # hide axis, ticks, ...
            #plt.title("Camera Capture")
            # show the plot!
            plt.show()

        else:
            # set the current frame as the data to show
            img.set_data(newimgR)
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
