import os
import pydicom
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt
from skimage.segmentation import chan_vese
import nibabel as nib
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter, zoom
import time
from cv2 import cv2

class DICOMProcessingFactory:
    def __init__(self, folder_path: str, start_idx: int, end_idx: int):
        self.folder_path = folder_path
        self.images = self._load_images(start_idx, end_idx)

    def _load_images(self, start_idx: int, end_idx: int) -> list:
        images = [
            (filename[:-4], pydicom.read_file(os.path.join(self.folder_path, filename)))
            for filename in os.listdir(self.folder_path) if filename.endswith('.dcm') and start_idx <= int(filename[:-4]) <= end_idx
        ]
        images.sort(key=lambda x: int(x[0]))
        return [img[1] for img in images]

    def run_functions(self, function_names, **kwargs):
        all_functions = self.get_all_functions()

        for func_name in function_names:
            if func_name in all_functions:
                print(f"Executing: {func_name}")
                all_functions[func_name](**kwargs.get(func_name, {}))
            else:
                print(f"Function {func_name} not recognized.")

    def invert_colors(self, invert_red=True, invert_green=True, invert_blue=True):
        img = self.images[0].pixel_array
        rgb_img = np.stack((img,) * 3, axis=-1)
        max_val = np.max(img)
        
        if invert_red:
            rgb_img[:, :, 0] = max_val - rgb_img[:, :, 0]
        if invert_green:
            rgb_img[:, :, 1] = max_val - rgb_img[:, :, 1]
        if invert_blue:
            rgb_img[:, :, 2] = max_val - rgb_img[:, :, 2]

        plt.imshow(rgb_img)
        plt.show()

    def adjust_brightness_contrast(self, brightness=0, contrast=1):
        img = self.images[0].pixel_array.astype(np.float32)
        img += brightness
        mean = np.mean(img)
        img = (img - mean) * contrast + mean
        plt.imshow(np.clip(img, 0, 255).astype(np.uint8), cmap=plt.cm.bone)
        plt.show()

    def dicom_contrast_correction(self, lower_bound, upper_bound):
        img = self.images[0].pixel_array.astype(np.int16)
        intercept = self.images[0].RescaleIntercept
        slope = self.images[0].RescaleSlope
        img = slope * img + intercept
        img = np.clip(img, lower_bound, upper_bound)
        img = ((img - intercept) / slope).astype(np.uint16)
        plt.imshow(img, cmap=plt.cm.bone)
        plt.show()
        img = img.astype(np.uint16)

        return img
    
    def windowing(self, level, width):
        img = self.images[0].pixel_array
        plt.imshow(np.clip(img, level - width // 2, level + width // 2), cmap=plt.cm.bone)
        plt.show()

    def resample(self, new_spacing=(1, 1)):
        img = self.images[0].pixel_array
        resample_factor = np.array(self.images[0].PixelSpacing) / new_spacing
        plt.imshow(zoom(img, resample_factor), cmap=plt.cm.bone)
        plt.show()

    def resample_image(self, img, new_shape=(256, 256, 10)):
        resample_factors = [ns/float(is_) for ns, is_ in zip(new_shape, img.shape)]
        return zoom(img, resample_factors)
    
    def noise_reduction(self, sigma=1):
        plt.imshow(gaussian_filter(self.images[0].pixel_array, sigma=sigma), cmap=plt.cm.bone)
        plt.show()

    def gamma_correction(self, img, gamma=1.0):
        if gamma == 0:
            print("Gamma value shouldn't be zero.")
            return
        normalized_img = img / 255.0
        corrected_img = np.power(normalized_img, gamma)
        corrected_img = (corrected_img * 255).astype(np.uint8)

        return corrected_img
    
    def padding_or_cropping(self, desired_shape=(512, 512)):
        img = self.images[0].pixel_array
        pad_before_x = (desired_shape[0] - img.shape[0]) // 2
        pad_after_x = desired_shape[0] - pad_before_x - img.shape[0]
        pad_before_y = (desired_shape[1] - img.shape[1]) // 2
        pad_after_y = desired_shape[1] - pad_before_y - img.shape[1]
        plt.imshow(np.pad(img, ((pad_before_x, pad_after_x), (pad_before_y, pad_after_y))), cmap=plt.cm.bone)
        plt.show()

    def normalization(self):
        img = self.images[0].pixel_array
        plt.imshow((img - np.min(img)) / (np.max(img) - np.min(img)), cmap=plt.cm.bone)
        plt.show()

    def get_all_functions(self):
        return {
            'aortic_windowing': self.aortic_windowing,
            'resample': self.resample,
            'noise_reduction': self.noise_reduction,
            'padding_or_cropping': self.padding_or_cropping,
            'normalization': self.normalization,
            'invert_colors': self.invert_colors
        }

    def save_all_images_as_png(self, mode='gamma', gamma=1.0, lower_bound=None, upper_bound=None, save_dir='.'):
        os.makedirs(save_dir, exist_ok=True)
        for idx, dicom_img in enumerate(self.images):
            resampled_img = self.resample_image(dicom_img.pixel_array, new_shape=(256, 256, 10))
            if mode == 'gamma':
                corrected_img = self.gamma_correction(resampled_img, gamma)
            elif mode == 'dicom_contrast':
                if lower_bound is None or upper_bound is None:
                    raise ValueError("Both lower_bound and upper_bound must be provided for dicom_contrast mode.")
                corrected_img = self.dicom_contrast_correction(resampled_img, lower_bound, upper_bound)
            else:
                raise ValueError(f"Unknown mode {mode}. Choose between 'gamma' and 'dicom_contrast'.")

            save_path = os.path.join(save_dir, f"{mode}_corrected_resampled_image_{idx}.png")
            plt.imsave(save_path, corrected_img, cmap=plt.cm.bone)
            print(f"Saved: {save_path}")

    def load_resize_save_images(self, target_size, output_directory):
        dcm_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.dcm')],
                           key=lambda x: int(x.split('.')[0]))

        middle_idx = len(dcm_files) // 2
        image1 = self.images[middle_idx - 1]
        image2 = self.images[middle_idx]
        image3 = self.images[middle_idx + 1]

        resized_image1 = cv2.resize(image1.pixel_array, target_size, interpolation=cv2.INTER_AREA).astype(np.uint8)
        resized_image2 = cv2.resize(image2.pixel_array, target_size, interpolation=cv2.INTER_AREA).astype(np.uint8)
        resized_image3 = cv2.resize(image3.pixel_array, target_size, interpolation=cv2.INTER_AREA).astype(np.uint8)

        images = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        images[..., 0] = resized_image1
        images[..., 1] = resized_image2
        images[..., 2] = resized_image3

        output_subdirectory = os.path.join(output_directory, self.folder_path.split(os.sep)[-1])
        os.makedirs(output_subdirectory, exist_ok=True)
        
        for i in range(images.shape[2]):
            filename = os.path.join(output_subdirectory, f"image_{i+1:03d}.png")
            plt.imsave(filename, images[:, :, i], cmap=plt.cm.bone)
            print(f"Saved: {filename}")

    @staticmethod
    def process_row(args):
        directory, target_size, output_directory = args
        processor = DICOMProcessingFactory(directory, 0, 1e9)  # assuming all images in the directory are to be processed
        processor.load_resize_save_images(target_size, output_directory)

def main():
    start_time = time.time()

    TRAIN_CSV = "rsna_data/train_series_meta.csv"
    data = pd.read_csv(TRAIN_CSV)

    target_size = (256, 256)
    output_directory = 'png_test_lyd/'
    directories = [
        os.path.join('rsna_data', 'train_images', str(int(row['patient_id'])), str(int(row['series_id'])))
        for _, row in data.iloc[100:200].iterrows()
    ]
    args_list = [(d, target_size, output_directory) for d in directories]

    with Pool(cpu_count()) as pool:
        pool.map(DICOMProcessingFactory.process_row, args_list)

    elapsed_time = time.time() - start_time
    print(f"Processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()