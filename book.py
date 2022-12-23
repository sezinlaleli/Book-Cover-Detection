# sezin laleli 191101040

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import math
from matplotlib import pyplot as plt


def calc_gaussian(windows_size, sigma):
    kernel = np.zeros((windows_size + 2, windows_size + 2))
    kernel[int((windows_size + 1) / 2)][int((windows_size + 1) / 2)] = 1
    kernel1 = cv2.GaussianBlur(kernel, (windows_size, windows_size), sigma)
    kernel2 = np.zeros((windows_size, windows_size))
    for i in range(1, windows_size + 1):
        for j in range(1, windows_size + 1):
            kernel2[i - 1][j - 1] = kernel1[i][j]
    return kernel2


# ssim is taking texture into account unlike mse
def ssim(img1, img2):
    # default variables
    sigma = 1.5
    windows_size = 11
    K1 = 0.01
    K2 = 0.03
    L = 255

    pad = windows_size // 2

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # 2D gaussian kernel
    window = calc_gaussian(windows_size, sigma)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    x = int(windows_size // 2)
    mu1 = cv2.filter2D(img1, -1, window)[x:-x, x:-x]
    mu2 = cv2.filter2D(img2, -1, window)[x:-x, x:-x]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[x:-x, x:-x] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[x:-x, x:-x] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[x:-x, x:-x] - mu1_mu2

    # size: size(img1) - size(window) + 1.
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def sift_features(image):
    sift_feature_limit = 1000
    sift = cv2.SIFT_create(sift_feature_limit)
    key_point, descriptors = sift.detectAndCompute(image, None)
    return key_point, descriptors


data = '/Science-Fiction-Fantasy-Horror'

# I will collect the paths of files in the dataset in a list
paths = []
for root, directories, files in os.walk(data):
    for file in files:
        paths.append((os.path.join(root, file)))

# we need to organize the data we have
# opencv only returns nNoneType object when it can't read an image such as gifs
# therefore we have to remove them before starting calculations
images = []
max_row = 0
max_col = 0
for path in paths:
    img = cv2.imread(path)
    if img is None:
        continue
    else:
        r, c = img.shape[:2]
        images.append(img)
        if r > max_row:
            max_row = r
        if c > max_col:
            max_col = c

# randomly choosing test image from all images and creating a file that contains them
# these steps are used to simulate a real-time image
# dst_dir = '/test_images'
# for j in range(50):
#    img_name = paths[random.randint(0, len(paths) - 1)]
#    for cpy in glob.iglob(os.path.join(data, img_name)):
#        shutil.copy(cpy, dst_dir)

data_test = '/test_images'

# I will collect the paths of files in the test set in a list
paths_test = []
for root, directories, files in os.walk(data_test):
    for file in files:
        paths_test.append((os.path.join(root, file)))

images_test = []
for path in paths_test:
    img = cv2.imread(path)
    if img is None:
        continue
    else:
        r, c = img.shape[:2]
        images_test.append(img)
        if r > max_row:
            max_row = r
        if c > max_col:
            max_col = c


def match(dst):
    #
    # 1st match -> correlation between histograms
    #

    # opencv reads images in BGR format
    # calculating rgb histogram of train data
    img_hist = []  # img_hist = {image, image_hist, image_path}
    color = ('b', 'g', 'r')
    for j in range(len(images)):
        t = cv2.cvtColor(images[j], cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([t], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None)
        img_hist.append((images[j], hist, paths[j]))

    # calculate histogram of the image that will be tested
    t = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    dst_hist = cv2.calcHist([t], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    dst_hist = cv2.normalize(dst_hist, None)

    # my train data is around 1k image if it was higher, then I would make the threshold higher such as 0.9
    correlation_threshold = 0.25
    matches = []  # matches = {comp_corr, image, image_path}
    for i in range(len(img_hist)):
        comp = cv2.compareHist(dst_hist, img_hist[i][1], cv2.HISTCMP_CORREL)
        if comp > correlation_threshold:
            matches.append((comp, img_hist[i][0], img_hist[i][2]))
    matches.sort(key=lambda x: x[0], reverse=True)

    #
    # 2nd match -> SSIM
    #

    # adding padding to all images for calculations in np
    rt, ct = dst.shape[:2]
    global max_row
    global max_col
    if max_row < rt:
        max_row = rt
    if max_col < ct:
        max_col = ct

    dst_pad = cv2.copyMakeBorder(dst, math.floor((max_row - rt) / 2), math.ceil((max_row - rt) / 2),
                                 math.floor((max_col - ct) / 2),
                                 math.ceil((max_col - ct) / 2), cv2.BORDER_REPLICATE, None, value=0)

    matches2 = []  # matches2 = {ssim_score, image, image_path}

    # based on my observation on ssim results
    similarity_index = 0.05

    for i in range(len(matches)):
        rd, cd = matches[i][1].shape[:2]
        img_pad = cv2.copyMakeBorder(matches[i][1], math.floor((max_row - rd) / 2), math.ceil((max_row - rd) / 2),
                                     math.floor((max_col - cd) / 2),
                                     math.ceil((max_col - cd) / 2), cv2.BORDER_REPLICATE, None, value=0)
        ssim_score = ssim(dst_pad, img_pad)
        if ssim_score > similarity_index:
            matches2.append((ssim_score, matches[i][1], matches[i][2]))

    matches2.sort(key=lambda x: x[0], reverse=True)

    # setting a limit as top 50 matches
    match_limit = 50
    matches2 = matches2[:match_limit]

    #
    # 3rd match -> SIFT
    #

    # because of the large number of features involved, searching for matches is computationally intensive
    # that's why we sifted through the pictures until we got to this step

    lowe_ratio = 0.75  # around ideal ratio

    # FLANN matcher
    # FLANN - Fast Library for Approximate Nearest Neighbors

    prediction_count = 3

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    last_matches = []  # last_matches = {match_count, match_path}
    flann_matches = []  # flann_matches = {match_count, match_path}
    train_img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    train_kp, train_des = sift_features(train_img)

    for i in range(len(matches2)):
        matches_count = 0
        match_path = matches2[i][2]
        match_img = cv2.imread(match_path)
        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        m_kp, m_des = sift_features(match_img)
        match_count = flann.knnMatch(train_des, m_des, k=2)  # calculating number of feature matches using FLAAN
        # ratio query as per David G. Lowe's paper
        for x, (m, n) in enumerate(match_count):
            if m.distance < lowe_ratio * n.distance:
                matches_count += 1
        flann_matches.append((matches_count, match_path))
    flann_matches.sort(key=lambda x: x[0], reverse=True)
    last_matches.append(flann_matches[:prediction_count])
    return last_matches

match_result = []
expected_result = []

lambda1 = 0.2
for i in range(len(paths_test)):
    expected_result.append(paths_test[i])
    test = images_test[i]
    mat_shear = np.array([[1, lambda1, 0], [lambda1, 1, 0], [0, 0, 1]])
    dst = ndi.affine_transform(test, mat_shear)
    match_result.append(match(dst))


tp = 0
for i in range(len(paths_test)):
    str1 = paths_test[i][paths_test[i].rfind('\\') + 1:len(paths_test[i])]
    str2 = match_result[i][0][0][1][match_result[i][0][0][1].rfind('\\') + 1:len(match_result[i][0][0][1])]
    if str2 == str1:
        tp += 1
print(tp)

# create figure
# fig = plt.figure(figsize=(10, 7))
#
# t = cv2.imread(paths_test[2])
# mat_shear = np.array([[1, lambda1, 0], [lambda1, 1, 0], [0, 0, 1]])
# d = ndi.affine_transform(t, mat_shear)
# d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
#
# fig.add_subplot(1, 2, 1)
# plt.imshow(d,)
# plt.axis('off')
# plt.title('test')
#
# r = cv2.imread(match_result[2][0][0][1])
# r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
#
# fig.add_subplot(1, 2, 2)
# plt.imshow(r,)
# plt.axis('off')
# plt.title('result')
#
# plt.show()
