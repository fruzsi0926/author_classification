import os
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
import math

class DatasetGenerator():
    def __init__(self, img_folder=None):

        #get img_path for all img in the folder
        self.img_paths = []

        for path, _, files in os.walk(img_folder, topdown=True):
            for name in files:
                self.img_paths.append((f"{path}/{name}").replace("\\","/"))


    def read(self, image_path=None, idx=None):
            if image_path is not None:
                return cv2.imread(image_path)
            if idx is not None:
                return cv2.imread(self.img_paths[idx])

    def show(self, img, gray=True, figsize=(15,20)): 
            plt.figure(figsize=figsize)
            if gray:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.show()
    
    def show_points(self, img, point_list, gray=False):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for point in point_list:
            cv2.circle(img,  (point[1], point[0]), radius=5, color=(255,0,0), thickness=4)
        self.show(img, gray)

    def show_crop(self, input_image, img_style="processed"):
        desired_size=(100,200)

        # kép előfeldolgozása és maximum pontok kinyerése
        image_gray_processed, points, peaks_list = self.preprocessing(input_image)

        if img_style == "processed":
            image=image_gray_processed
        elif img_style == "original":
            image=input_image
        else:
            raise "Wrong color only RGB or gray is choosable."
            
        # a kivágás mérete 
        h = self.get_mean_line_distance(peaks_list)//2
        w = int(desired_size[1]*(h/desired_size[0]))//2

        img_shape = image.shape
        img_idx = 0
        for idx, point in enumerate(points):
            if idx % 10 == 0:
                img_idx += 1
                x = point[0]
                y = point[1]
                if (x>h+1) and (y>w+1) and (x<(img_shape[0]-h-1)) and (y < (img_shape[1]-w-1)):
                    # if RGB image
                    if len(image.shape) > 2:
                        crop_img = image[x-h:x+h,y-w:y+w,:]
                        crop_img = self.resize_cropped_img_to_unified_size(crop_img, desired_size)
                    # if 2D grayscale
                    else:
                        crop_img = image[x-h:x+h,y-w:y+w]
                        crop_img = self.resize_cropped_img_to_unified_size(crop_img, desired_size)
                    self.show(crop_img,True, (5,3))
                    print(crop_img.shape)
                    if img_idx > 15:
                        break

    def preprocessing(self, img):
        #IDE KELL MAJD A VÉGSŐ PREPROCESSING
        ##############################################
        if len(img.shape) > 2 and img.shape[2] ==3:
            denoising = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            denoising = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        gray = cv2.cvtColor(denoising, cv2.COLOR_BGR2GRAY)

        processed_image = cv2.adaptiveThreshold(gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        ###########################################xx

        point_list, peaks_list = self.conv_and_find_maximas(processed_image)

        return processed_image, point_list, peaks_list
    
    def generate_train_dataset(
        self,
        img_style="processed",
        desired_size=(100,200)
        ):
        output_folder_name = f"cropped_{img_style}"
        if not os.path.exists(f"dataset/ready_for_train/{output_folder_name}"):
            os.makedirs(f"dataset/ready_for_train/{output_folder_name}")

        
        #egyesével iteráljunk végig a képen és a nevén(label/szerző)
        past_label=""
        for img_path in tqdm(self.img_paths, total=len(self.img_paths)):
            # kép betöltése és szerző elemntése labelként
            input_image = cv2.imread(img_path)
            label = img_path.split("/")[2]
            if label != past_label:
                img_idx = 0
                print(f"Processing images for ({label}) class")
            past_label = label

            # kép előfeldolgozása és maximum pontok kinyerése
            if input_image is None:
                print(f"Found wrong image: {img_path}")
                continue
            image_gray_processed, points, peaks_list = self.preprocessing(input_image)

            if img_style == "processed":
                image=image_gray_processed
            elif img_style == "original":
                image=input_image
            else:
                raise "Wrong img_style only (processed) or (original) is choosable."

            # a kivágás mérete
            h = self.get_mean_line_distance(peaks_list)//2
            w = int(desired_size[1]*(h/desired_size[0]))//2

            # labelenként egy mappa
            save_folder = f"dataset/ready_for_train/{output_folder_name}/{label}"
            if not os.path.exists(save_folder):
                os.makedirs(f"dataset/ready_for_train/{output_folder_name}/{label}")

            cropped_images = []
            img_shape = image.shape
            for point in tqdm(points, total=len(points)):
                x = point[0]
                y = point[1]
                if (x>h+1) and (y>w+1) and (x<(img_shape[0]-h-1)) and (y < (img_shape[1]-w-1)):
                    # if RGB image
                    if len(image.shape) > 2:
                        crop_img = image[x-h:x+h,y-w:y+w,:]
                        crop_img = self.resize_cropped_img_to_unified_size(crop_img, desired_size)
                    # if 2D grayscale
                    else:
                        crop_img = image[x-h:x+h,y-w:y+w]
                        crop_img = self.resize_cropped_img_to_unified_size(crop_img, desired_size)
                    name = f"{save_folder}/{img_idx}.png"
                    cv2.imwrite(name, crop_img)
                    cropped_images.append(crop_img)
                    img_idx += 1
    


    def conv_and_find_maximas(self, image):
        kernel_h, kernel_w = 20, 50
        img_h, img_w = image.shape
        tr = 0.3
        step_w = 50

        # do special convolution
        h = range(img_h-kernel_h)
        w = range(0, img_w-kernel_w, step_w)
        result = np.zeros((img_h,img_w))

        for y in w:
            for x in h:
                window = image[x:x+kernel_h,y:y+kernel_w]

                white_ratio = np.sum(window==255)/(kernel_h*kernel_w)
                result[x+kernel_h//2,y:y+step_w] = white_ratio

        # find local maximas in all columns
        point_list = []
        peaks_list = []
        local_max_counter = 0
        for spec_w in w:
            x = result[:,spec_w]
            peaks, _ = find_peaks(x, height=0.17,distance=30)
            peaks_list.append(peaks)
            for peak in peaks:
                point_list.append((peak,spec_w+step_w//2))
                local_max_counter += 1
        return point_list, peaks_list


    def get_mean_line_distance(self, peaks_list, verbose=False):
        all_diff = np.array([])

        # extract differences
        for column in peaks_list:
            difference = np.diff(column)
            all_diff = np.append(all_diff, difference)
        
        # filter outliers (too big distances (and a bit of low distances))
        sorted_differences = np.sort(all_diff)
        nr_of_diffs = len(sorted_differences)
        limit_low, limit_high = math.ceil(nr_of_diffs*0.2), math.floor(nr_of_diffs*0.8)
        filtered_differences = sorted_differences[limit_low:limit_high]
        if verbose:
            print(f"Filtered {limit_low}/{nr_of_diffs} too low and {nr_of_diffs-limit_high}/{nr_of_diffs} too high outliers.")
            print(f"Limit low: {filtered_differences[0]}, Limit_low: {filtered_differences[-1]}")
        mean_line_distance = int(np.mean(filtered_differences)*1.1)
        if verbose:
            print(f"Based on the remained {filtered_differences.shape[0]} line differences, "
            f"the mean line distance: {mean_line_distance} pixels")

        return mean_line_distance

    def resize_cropped_img_to_unified_size(self, img, desired_size=(50,100)):
        return cv2.resize(img, (desired_size[1],desired_size[0]))

    def generate_test_dataset(
        self,
        img_style="processed",
        desired_size=(100,200),
        temp_folder_name=None
        ):
        assert temp_folder_name is not None, "Pass temp folder name please."

        #egyesével iteráljunk végig a képen és a nevén(label/szerző)
        past_label=""
        for index, img_path in tqdm(enumerate(self.img_paths), total=len(self.img_paths)):
            # kép betöltése és szerző elemntése labelként
            input_image = cv2.imread(img_path)
            label = f"test_{index}"
            if label != past_label:
                print(f"Processing ({label}) image")
            past_label = label

            # kép előfeldolgozása és maximum pontok kinyerése
            image_gray_processed, points, peaks_list = self.preprocessing(input_image)

            if img_style == "processed":
                image=image_gray_processed
            elif img_style == "original":
                image=input_image
            else:
                raise "Wrong img_style only (processed) or (original) is choosable."

            # a kivágás mérete
            h = self.get_mean_line_distance(peaks_list)//2
            w = int(desired_size[1]*(h/desired_size[0]))//2

            # labelenként egy mappa
            save_folder = f"{temp_folder_name}/{label}"
            if not os.path.exists(save_folder):
                os.makedirs(f"{temp_folder_name}/{label}")

            cropped_images = []
            img_shape = image.shape
            img_idx = 0
            for point in tqdm(points, total=len(points)):
                x = point[0]
                y = point[1]
                if (x>h+1) and (y>w+1) and (x<(img_shape[0]-h-1)) and (y < (img_shape[1]-w-1)):
                    # if RGB image
                    if len(image.shape) > 2:
                        crop_img = image[x-h:x+h,y-w:y+w,:]
                        crop_img = self.resize_cropped_img_to_unified_size(crop_img, desired_size)
                    # if 2D grayscale
                    else:
                        crop_img = image[x-h:x+h,y-w:y+w]
                        crop_img = self.resize_cropped_img_to_unified_size(crop_img, desired_size)
                    name = f"{save_folder}/{img_idx}.png"
                    cv2.imwrite(name, crop_img)
                    cropped_images.append(crop_img)
                    img_idx += 1

