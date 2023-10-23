import os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import pydicom
import numpy as np
from IPython.display import HTML
import matplotlib.animation as animation
from tqdm import tqdm

DS_RATE = 2


# class DriveManager:
    

#     @staticmethod
#     def mount_and_navigate_to_dir(directory_path="/content/drive/MyDrive/rsna_data/"):
#         ROOT = "/content/drive"
#         drive.mount(ROOT, force_remount=True)
#         os.chdir(directory_path)
#         print(f"Current directory: {os.getcwd()}")


class DataFrameAnalyzer:
    

    @staticmethod
    def dataframe_info(df):
        
        print("Shape:", df.shape)
        print("\nInfo:")
        print(df.info())
        print("\nDistinct count and unique values for each column:")
        for col in df.columns:
            print(f"{col} (Distinct Count: {df[col].nunique()}): {df[col].unique()}")
        print("\nNull values for each column:")
        for col in df.columns:
            print(f"{col}: {df[col].isnull().sum()}")
        print("\nDescriptive Statistics:")
        print(df.describe(include='all'))
        for col in df.columns:
            print(f"\nValue counts for column {col}:")
            print(df[col].value_counts())


class NIFTIHandler:
    

    @staticmethod
    def load_all_nii_files():
        
        nii_files = [f for f in os.listdir("segmentations") if f.endswith(('.nii', '.nii.gz'))]
        return [nib.load(os.path.join("segmentations", f)) for f in nii_files]

    @staticmethod
    def plot_nii_images(nii_images, num_to_plot=2):
        
        for nii in nii_images[:num_to_plot]:
            data = nii.get_fdata()
            slice_idx = data.shape[2] // 2
            plt.imshow(data[:, :, slice_idx], cmap='gray')
            plt.axis('off')
            plt.show()


class DICOMAnimation:
    

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = self._load_images()

    def _load_single_image(self, filename):
        
        return pydicom.read_file(os.path.join(self.folder_path, filename)).pixel_array

    def _load_images(self):
        
        dicom_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.dcm')])
        with ProcessPoolExecutor() as executor:
            images_data = list(tqdm(executor.map(self._load_single_image, dicom_files)))
        return images_data

    def show(self):
        
        fig, ax = plt.subplots()
        im = ax.imshow(self.images[0], cmap=plt.cm.bone)
        ax.axis('off')

        def update(i):
            im.set_array(self.images[i])
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=range(len(self.images)), repeat=True, blit=True)
        return HTML(ani.to_jshtml())

class Medical3DImageVisualizer:
    def __init__(self, downsample_rate=1):
        self.downsample_rate = downsample_rate

    def _load_dicom_images(self, folder):
        
        filenames = sorted([int(f.split('.')[0]) for f in os.listdir(folder)])
        return [os.path.join(folder, f"{filename}.dcm") for filename in filenames]

    def create_3D_scans(self, folder):
        
        filepaths = self._load_dicom_images(folder)
        volume = [self._process_dicom_image(pydicom.dcmread(fp)) for fp in tqdm(filepaths[::self.downsample_rate])]
        return np.stack(volume, axis=0)

    def _process_dicom_image(self, ds):
        
        intercept, slope = float(ds.RescaleIntercept), float(ds.RescaleSlope)
        center, width = int(ds.WindowCenter), int(ds.WindowWidth)
        image = (ds.pixel_array * slope) + intercept
        image = np.clip(image, center - width // 2, center + width // 2)
        image = (image / np.max(image) * 255).astype(np.int16)
        return image[::self.downsample_rate, ::self.downsample_rate]

    def create_3D_segmentations(self, filepath):
        
        img = nib.load(filepath).get_fdata()
        img = np.rot90(img, 3, (0, 2))
        return img[::self.downsample_rate, ::self.downsample_rate, ::self.downsample_rate]

    def plot_image_with_seg(self, volume, volume_seg=[], orientation='Coronal', num_subplots=20):
        
        plot_mask = len(volume_seg) > 0
        volume, volume_seg = self._orient_volume(volume, orientation, volume_seg)

        slices = np.linspace(0, volume.shape[0] - 1, num_subplots).astype(np.int16)
        self._plot_slices(volume, volume_seg, slices, plot_mask)

    def _orient_volume(self, volume, orientation, volume_seg):
        
        if orientation == 'Coronal':
            volume = volume.transpose([1, 0, 2])
            volume_seg = volume_seg.transpose([1, 0, 2]) if volume_seg else []
        elif orientation == 'Sagittal':
            volume = volume.transpose([2, 0, 1])
            volume_seg = volume_seg.transpose([2, 0, 1]) if volume_seg else []
        return volume, volume_seg

    def _plot_slices(self, volume, volume_seg, slices, plot_mask):
        
        rows = max([np.floor(np.sqrt(len(slices))).astype(int) - 2, 1])
        cols = np.ceil(len(slices) / rows).astype(int)
        fig, ax = plt.subplots(rows, cols, figsize=(cols * 2, rows * 4))
        fig.tight_layout(h_pad=0.01, w_pad=0)
        ax = ax.ravel()
        for this_ax in ax:
            this_ax.axis('off')

        for counter, idx in enumerate(slices):
            plt.subplot(rows, cols, counter + 1)
            plt.imshow(volume[idx], cmap='bone', interpolation='bicubic')
            if plot_mask:
                plt.imshow(volume_seg[idx], alpha=0.4, cmap='viridis')
            plt.axis('off')
        plt.show()


class DataHandler:
    

    def __init__(self, downsample_rate=1):
        self.visualizer = Medical3DImageVisualizer(downsample_rate)

    def load_and_display_dicom_images(self, folder_path: str):
        dicom_animation = DICOMAnimation(folder_path)
        return dicom_animation.show()

    def load_and_display_3D_images(self, folder_path: str, nifti_file: str, orientation='Coronal'):
        volume = self.visualizer.create_3D_scans(folder_path)
        volume_seg = self.visualizer.create_3D_segmentations(nifti_file)
        self.visualizer.plot_image_with_seg(volume, volume_seg, orientation)


if __name__ == "__main__":
    # manager = DriveManager()
    # manager.mount_and_navigate_to_dir("/content/drive/MyDrive/rsna_data/")
    
    data_handler = DataHandler(DS_RATE)
    
    print("\nLoading and Displaying DICOM Images:")
    data_handler.load_and_display_dicom_images("")
    
    print("\nLoading and Displaying 3D Images:")
    data_handler.load_and_display_3D_images("", "")