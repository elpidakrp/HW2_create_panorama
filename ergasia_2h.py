import os
import cv2 as cv
import numpy as np

x_img = 640
y_img = 480
outer_x = x_img * 5
outer_y = y_img * 3

def find_matches(descriptors_1, descriptors_2):
    print('Finding matches...')
    candidates = []
    matches = []

    for i1 in range(descriptors_1.shape[0]):
        some_row = descriptors_1[i1, :]
        diff = descriptors_2 - some_row
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        i2 = np.argmin(distances)
        candidates.append(i2)
    for i2 in range(descriptors_2.shape[0]):
        some_row = descriptors_2[i2, :]
        diff = descriptors_1 - some_row
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)
        # print('Crosschecking the matches...')
        i1 = np.argmin(distances)
        if candidates[int(i1)] == i2:
            matches.append(cv.DMatch(i1, i2, distances[i1]))
    print('Done!')
    return matches

def shrink_and_center_image (img):
    print('Shrinking image...')
    center = np.array([[1, 0, x_img * 2], [0, 1, y_img]], dtype=np.float32)
    img = cv.resize(img, (x_img, x_img))
    print('Image shrinked now centering...')
    img = cv.warpAffine(img, center, (outer_x, outer_y))
    print('Done!')
    return img

pic_folder = ['folder_with pictures']

print('Folder with pictures spoted! /'
      '\n Initializing img_array, keypoints, descriptors...')

sift_or_surf = 'sift'

if sift_or_surf == 'sift':
    method = cv.xfeatures2d_SIFT.create()
else:
    method = cv.xfeatures2d_SURF.create()

print('Chose method ' + sift_or_surf)

img_array = []
keypoints = []
descriptors = []


print('Done!')

for folder in pic_folder:
    files = os.listdir(folder)
    i = 0
    for file in files:
        path = os.path.join(folder, file)
        img = cv.imread(path)
        img = shrink_and_center_image(img)
        print('Appending image 0'+str(i)+', keypoints and descriptors with '+sift_or_surf+'...')
        img_array.append(img)
        keypoints.append(method.detectAndCompute(img, None)[0])
        descriptors.append(method.detectAndCompute(img, None))
        i += 1

print('Done! \n'
      'Stiching right images...')

img_new = img_array[0].copy()
print(' Starting with image 00...')
for i in range(2):
    print(' Matching image 0'+str(i+1)+'...')
    keypoints_new = method.detect(img_new)
    descriptors_new = method.compute(img_new, keypoints_new)

    matches = find_matches(descriptors_new[1], descriptors[i + 1][1])
    print(' Showing matching result...')
    d_img = cv.drawMatches(img_new, descriptors_new[0], img_array[i + 1], descriptors[i + 1][0], matches, None)
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', d_img)
    cv.waitKey(0)

    img_pt1 = []
    img_pt2 = []
    for x in matches:
        img_pt1.append(keypoints_new[x.queryIdx].pt)
        img_pt2.append(keypoints[i + 1][x.trainIdx].pt)
    img_pt1 = np.array(img_pt1)
    img_pt2 = np.array(img_pt2)

    print(' Finding homography and applying it to the right image...')
    M, _ = cv.findHomography(img_pt1, img_pt2, cv.RANSAC)
    img_new = cv.warpPerspective(img_new, M, (outer_x, outer_y))
    print(' Stiching the images...')
    img_new[:, x_img * 2: x_img * 3] = img_array[i + 1][:, x_img * 2: x_img * 3]
    cv.namedWindow('main', cv.WINDOW_NORMAL)
    cv.imshow('main', img_new)
    cv.waitKey(0)

print('Done with right images!\n'
      'Now stitching left images...')

image_new = img_array[4].copy()
print(' Starting with image 04...')
for j in range(3, 1,-1):
    print(' Matching image 0'+str(j)+'...')
    kp_new = method.detect(image_new)
    desc_new = method.compute(image_new, kp_new)
    matches = find_matches(desc_new[1], descriptors[j][1])
    print(' Showing matching result...')
    d_img = cv.drawMatches(image_new, desc_new[0], img_array[j], descriptors[j][0], matches, None)
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', d_img)
    cv.waitKey(0)

    img_pt1 = []
    img_pt2 = []
    for x in matches:
        img_pt1.append(kp_new[x.queryIdx].pt)
        img_pt2.append(keypoints[j][x.trainIdx].pt)
    img_pt1 = np.array(img_pt1)
    img_pt2 = np.array(img_pt2)

    print(' Finding homography and applying it to the left image...')
    M, _ = cv.findHomography(img_pt1, img_pt2, cv.RANSAC)
    left_image = cv.warpPerspective(image_new, M, (outer_x, outer_y))
    print(' Stiching the images...')
    left_image[:, x_img * 2 : x_img*3] = img_array[j][:, x_img * 2 : x_img*3]
    image_new = left_image.copy()
    cv.namedWindow('main', cv.WINDOW_NORMAL)
    cv.imshow('main', image_new)
    cv.waitKey(0)

print('Done with left images!\n'
      'Stitching all together...')

img_new[:, 0: x_img*2] = image_new[:, 0: x_img*2]
print('Done!\n'
      'Showing the result...')
cv.namedWindow('main', cv.WINDOW_NORMAL)
cv.imshow('main', img_new)
cv.waitKey(0)

cv.imwrite('my_panorama_sift.jpg', img_new)
