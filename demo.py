from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# saved data file names
input_file = 'input.data'
output_file = 'output.data'
collage_file = 'collage.png'
test_img = 'images/edge15.png'

# hard-coded mnist size, which is the input to the pre-trained network
size_mnist = (28, 28)
n_class = 10  # 0-9 and a class for nothing

# handpicked position, from scaled image (scaled so that each row has 28 pixels in height, just like mnist)
position0 = ('edge82', 28, 29)
position1 = ('edge33', 28, 104)
position2 = ('edge33', 28, 122)
position3 = ('edge82', 0, 29)
position4 = ('edge20', 28, 122)
position5 = ('edge21', 28, 122)
position6 = ('edge22', 28, 122)
position7 = ('edge23', 28, 122)
position8 = ('edge24', 28, 122)
position9 = ('edge25', 28, 122)
position10 = ('edge82', 0, 0)

# So ugly I'm gonna puke
all_position = [position0, position1, position2, position3, position4, position5, position6, position7, position8,
                position9]


# open image as gray scale
def crop_image(pos, offset=0):
    file, i, j = pos
    j = j + offset
    img = Image.open('images/' + file + '.png').convert('L')

    # resize to make the new heigtht double that of mnist, so we don't have to convolute in y
    width, height = img.size
    new_height = size_mnist[0] * 2
    ratio = new_height / height
    new_width = np.floor(ratio * width)
    img = img.resize((np.int(new_width), np.int(new_height)), Image.ANTIALIAS)
    x = np.asarray(img)

    # subsample to mnist size
    x = x[i:i + size_mnist[0], j:j + size_mnist[1]]
    return x


# get contour from a grayscale image
def get_contour(img):
    # img: already inverted gray scale image
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    #################      Now finding Contours         ###################

    imc, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return thresh, contours


def generate_samples(n_row=80, n_col=n_class):

    # Since standard mnist model predicts so shit, this is the data
    # we will use with perturbation to create our own training data
    all_x = [crop_image(pos) for pos in all_position]

    # Create a synthetic image by random sampling
    height, width = all_x[0].shape
    large_img = np.tile(all_x[0], (n_row, n_col))

    roi_shape = (10, 10)
    roi_size = roi_shape[0]*roi_shape[1]
    samples = np.empty((0, roi_size))
    responses = []

    all_h = []
    all_area = []

    for ii in range(n_row):
        # jj == number
        for jj in range(n_col):
            # randomly perturb the image
            angle = np.random.randn()*10
            small_img = Image.fromarray(np.invert(all_x[jj]))
            small_img = small_img.rotate(angle)
            small_img = np.invert(np.asarray(small_img))

            # This one requires uninverted
            thresh, contours = get_contour(small_img)
            # find the largest contour
            ss = [cv2.contourArea(cnt) for cnt in contours]
            [x, y, w, h] = cv2.boundingRect(contours[np.argmax(ss)])

            # cv2.rectangle(small_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = thresh[y:y + h, x:x + w]
            small_roi = cv2.resize(roi, roi_shape)
            cv2.imshow('norm', small_img)

            large_img[ii*height:(ii+1)*height, jj*width:(jj+1)*width] = small_img

            samples = np.append(samples, small_roi.reshape((1,roi_size)), 0)
            responses.append(jj)

            all_h.append(h)
            all_area.append(cv2.contourArea(contours[np.argmax(ss)]))

    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))

    np.savetxt(input_file, samples)
    np.savetxt(output_file,responses)

    # Plot
    f = plt.figure(figsize=(12, 8))
    large_img = Image.fromarray(large_img)
    plt.imshow(large_img)
    large_img.save(collage_file)

    print("generated {:d} x {:d} samples, which can be visualised in {:s}".format(n_row, n_col, collage_file))


def train():
    # KNearest neighbour classification on a flattened vector of the shrinked contour content
    samples = np.loadtxt(input_file, np.float32)
    responses = np.loadtxt(output_file, np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    return model

def test(model, test_img=test_img):
    im = cv2.imread(test_img)
    height, width, depth = im.shape
    scale = size_mnist[1]*2/height
    # Scale to training data size (roughtly 28pixels as height per digit)
    im = cv2.resize(im,(int(width*scale), int(height*scale)))
    print(im.shape)

    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    img2, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('img2', img2)

    for cnt in contours:
        if cv2.contourArea(cnt)>20:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>14:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                # retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                string = str(int((results[0][0])))
                cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

    cv2.imshow('im',im)
    cv2.imshow('out',out)
    cv2.waitKey(0)

if __name__ == "__main__":
    generate_samples()
    model = train()
    test(model)

