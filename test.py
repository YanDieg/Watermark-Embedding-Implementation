import copy
import math
import os
import random

import cv2
import numpy as np


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if (width > image_size[0]):
        width = image_size[0]

    if (height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def prepare_test(src, outfolder):

    #print("prepare_test")
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)

    watermarked = cv2.imread(src, cv2.IMREAD_COLOR)
    _, src_name_ext = os.path.split(src)
    src_name, _ = os.path.splitext(src_name_ext)

    # generate rotated images
    for angle in [0.25, 0.5, 2, 5, 30, 45, 90]:
        outfilename = os.path.join(outfolder, "%s_rot%s.%s" %
                                   (src_name, str(angle).replace('.','_'), "png"))
        image_center = tuple(np.array(watermarked.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
        rotated = cv2.warpAffine(watermarked, rot_mat,
                                 watermarked.shape[1::-1], flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))

        cv2.imwrite(outfilename, rotated)
        #print("write:", outfilename)

    # generate compressed jpg images
    for quality in [50, 80]:
        outfilename = os.path.join(outfolder, "%s_jpg%d.%s" %
                                   (src_name, quality, "jpg"))
        cv2.imwrite(outfilename, watermarked, [
            cv2.IMWRITE_JPEG_QUALITY, quality])
        #print("write:", outfilename)

    for i in [3, 5, 7]:
        kernel = np.ones((i, i), np.float32) / i ** 2
        dst = cv2.filter2D(watermarked, -1, kernel)
        outfilename = os.path.join(outfolder, "%s_median%d.%s" %
                                   (src_name, i, "png"))
        cv2.imwrite(outfilename, dst)

    # generate scaled images 0.5
    scale = 0.5
    outfilename = os.path.join(outfolder, "%s_scale0_5.%s" %
                               (src_name, "png"))
    imgH, imgW, c = watermarked.shape
    scaled = cv2.resize(watermarked, (int(round(imgH * scale)), int(round(imgW * scale))),
                        interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(outfilename, scaled)
    #print("write:", outfilename)

    # generate scaled images 2
    scale = 2
    outfilename = os.path.join(outfolder, "%s_scale2.%s" %
                               (src_name, "png"))
    imgH, imgW, c = watermarked.shape
    scaled = cv2.resize(watermarked, (int(round(imgH * scale)), int(round(imgW * scale))),
                        interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(outfilename, scaled)
    #print("write:", outfilename)

    # generate cropping image
    for factor in [50, 75]:
        outfilename = os.path.join(outfolder, "%s_crop%d.%s" %
                                   (src_name, factor, "png"))

        imgH, imgW, c = watermarked.shape

        y = int(round(imgH * (100 - factor) * 0.01 / 2))
        x = int(round(imgW * (100 - factor) * 0.01 / 2))
        h = int(round(y + imgH * 0.01 * (factor)))
        w = int(round(x + imgW * 0.01 * (factor)))
        cropped = cv2.copyMakeBorder(watermarked[x:w, y:h], y, y, x, x, cv2.BORDER_CONSTANT)

        cv2.imwrite(outfilename, cropped)
        #print("write:", outfilename)

    # generate rotation crop 0.25

    outfilename = os.path.join(outfolder, "%s_rotation_crop0_25.%s" %
                               (src_name, "png"))
    angle = 0.25
    image = watermarked
    image_height, image_width = image.shape[0:2]

    image_rotated = rotate_image(image, angle)
    cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(angle)
        )
    )
    y, x, c = cropped.shape
    cropped_final = cv2.copyMakeBorder(cropped, int(round((imgH - y)/2)), imgH - int(round((imgH - y)/2)) - y, int(round((imgW - x)/2)), imgW - y - int(round((imgW - x)/2)), cv2.BORDER_CONSTANT)

    cv2.imwrite(outfilename, cropped_final)
    #print("write:", outfilename)

    # generate rotation scale 0.25
    outfilename = os.path.join(outfolder, "%s_rotation_scale0_25.%s" %
                               (src_name, "png"))
    imgH, imgW, c = watermarked.shape
    scaled = cv2.resize(cropped, (imgH, imgW), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(outfilename, scaled)
    #print("write:", outfilename)

    # generate rotation crop -0.25
    angle = -0.25
    image = watermarked
    image_height, image_width = image.shape[0:2]

    image_rotated = rotate_image(image, angle)
    cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(angle)
        )
    )
    y, x, c = cropped.shape
    cropped_final = cv2.copyMakeBorder(cropped, int(round((imgH - y)/2)), imgH - int(round((imgH - y)/2)) - y, int(round((imgW - x)/2)), imgW - y - int(round((imgW - x)/2)), cv2.BORDER_CONSTANT)

    outfilename = os.path.join(outfolder, "%s_rotation_crop-0_25.%s" %
                               (src_name, "png"))
    cv2.imwrite(outfilename, cropped_final)
    #print("write:", outfilename)

    # generate rotation scale -0.25
    outfilename = os.path.join(outfolder, "%s_rotation_scale-0_25.%s" %
                               (src_name, "png"))
    imgH, imgW, c = watermarked.shape
    scaled = cv2.resize(cropped, (imgH, imgW), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(outfilename, scaled)
    #print("write:", outfilename)

    # generate removed lines image
    for i in [10, 50, 70, 100]:
        dst = copy.deepcopy(watermarked)
        j = 0
        indextoremove = []

        while j < 512:
            removedline = random.sample(range(j, j + i + 1), 1)
            if removedline[0] > 511:
                removedline[0] = 511

            indextoremove.append(removedline[0])
            j = j + i

        count = 0;
        for k in indextoremove:
            k = k - count;
            dst = np.delete(dst, k, axis=0)
            dst = np.delete(dst, k, axis=1)
            count = count + 1;

        outfilename = os.path.join(outfolder, "%s_removedLines%d.%s" %
                                       (src_name, i, "png"))
        cv2.imwrite(outfilename, dst)
        #print("write:", outfilename)



# START
def launchtest():
    prepare_test("img/encodeOutput/imageWithWatermark.png", "img/decodeInput")

if __name__ == '__main__':
    launchtest()



