import os
import pydicom
import numpy as np
from PIL import Image


class BaseLoader:

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.data = self._load_data()

    def _load_data(self):
        raise NotImplementedError("Subclasses should implement this method!")

    def display_shapes(self):
        raise NotImplementedError("Subclasses should implement this method!")


class DICOMLoader(BaseLoader):

    def _load_data(self) -> list:
        image_files = [filename for filename in os.listdir(self.folder_path) if filename.endswith('.dcm')]
        print(f"Found {len(image_files)} DICOM files in {self.folder_path}")
        return [pydicom.read_file(os.path.join(self.folder_path, filename)) for filename in image_files]

    def display_shapes(self):
        for idx, img in enumerate(self.data):
            array = img.pixel_array
            self._display_shape(array, idx)

    @staticmethod
    def _display_shape(array, idx):
        if len(array.shape) == 2:  
            print(f"Image {idx + 1}: Width: {array.shape[1]}, Height: {array.shape[0]}, Depth: 1, Channels: 1")
        elif len(array.shape) == 3:  
            print(f"Image {idx + 1}: Width: {array.shape[1]}, Height: {array.shape[0]}, Depth: 1, Channels: {array.shape[2]}")
        else:  
            print(f"Image {idx + 1} has an unusual shape: {array.shape}")


class PNGLoader(BaseLoader):

    def _load_data(self) -> list:
        image_files = [filename for filename in os.listdir(self.folder_path) if filename.endswith('.png')]
        print(f"Found {len(image_files)} PNG files in {self.folder_path}")
        return [Image.open(os.path.join(self.folder_path, filename)) for filename in image_files]

    def display_shapes(self):
        for idx, img in enumerate(self.data):
            width, height = img.size
            if len(img.getbands()) == 1:  
                print(f"Image {idx + 1}: Width: {width}, Height: {height}, Depth: 1, Channels: 1")
            else:  
                print(f"Image {idx + 1}: Width: {width}, Height: {height}, Depth: 1, Channels: {len(img.getbands())}")


class NPYLoader(BaseLoader):

    def _load_data(self) -> list:
        array_files = [filename for filename in os.listdir(self.folder_path) if filename.endswith('.npy')]
        print(f"Found {len(array_files)} NPY files in {self.folder_path}")
        return [np.load(os.path.join(self.folder_path, filename)) for filename in array_files]

    def display_shapes(self):
        for idx, array in enumerate(self.data):
            self._display_shape(array, idx)

    @staticmethod
    def _display_shape(array, idx):
        height, width = array.shape[:2]
        if len(array.shape) == 2:  
            print(f"Array {idx + 1}: Width: {width}, Height: {height}, Depth: 1, Channels: 1")
        elif len(array.shape) == 3:  
            depth_or_channels = array.shape[2]
            print(f"Array {idx + 1}: Width: {width}, Height: {height}, Depth: {depth_or_channels}, Channels: {depth_or_channels}")
        else:  
            print(f"Array {idx + 1} has an unusual shape: {array.shape}")


if __name__ == '__main__':
    
    dicom_dir = "./path_to_dicom_folder"
    dicom_loader = DICOMLoader(dicom_dir)
    dicom_loader.display_shapes()

    png_dir = "./path_to_png_folder"
    png_loader = PNGLoader(png_dir)
    png_loader.display_shapes()

    npy_dir = "./path_to_npy_folder"
    npy_loader = NPYLoader(npy_dir)
    npy_loader.display_shapes()
