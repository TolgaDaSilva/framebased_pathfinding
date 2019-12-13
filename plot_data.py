import os
import cv2
import json
from utility import *


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

if __name__ == "__main__":
    radius = 300
    os.chdir('flat')
    dirs = [f for f in os.listdir() if not f.startswith('.')]
    j = 0
    for _,dir in enumerate(dirs):
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f))]
        if(j == 0):
            j += 1
            continue
        files = files[20:46]
        for i in range(0, len(files),2):
            img = cv2.imread(os.path.join(dir,files[i+1]), cv2.COLOR_RGB2BGR)
            with open(os.path.join(dir, files[i])) as json_file:
                set_labels(json.load(json_file))
            b,g,r = cv2.split(img)
            rgb_img = cv2.merge([r,g,b])
            gamma_img = adjust_gamma(rgb_img,2.0)
            p1,p2 = get_fov(0,0,90,90,60)
            create_nodes(0, 0, p1[0], p1[1], p2[0], p2[1])
            plt.subplot(1,2,1)
            plot_data()
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(gamma_img)
            plt.axis('off')
            plt.title(files[i+1])
            plt.show()
            plt.close()
