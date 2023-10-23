import gc
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras_core as keras
from matplotlib import pyplot as plt
import keras_cv
import pydicom
import cv2
import tqdm

class Config:
    SEED = 42
    IMAGE_SIZE = [256, 256]
    BATCH_SIZE = 16
    EPOCHS = 15
    TARGET_COLS = ["bowel_healthy", "bowel_injury", "extravasation_healthy",
                   "extravasation_injury", "kidney_healthy", "kidney_low",
                   "kidney_high", "liver_healthy", "liver_low", "liver_high",
                   "spleen_healthy", "spleen_low", "spleen_high"]
    AUTOTUNE = tf.data.AUTOTUNE
    BASE_PATH = "/content/drive/MyDrive/rsna_data"
    SAVE_DIR = os.path.join(BASE_PATH, "checkpoint")


class DataPreparation:

    def __init__(self, config: Config):
        self.config = config
        self.df = self._load_data()

    def _load_data(self):
        train_df = pd.read_csv(f"{self.config.BASE_PATH}/train.csv")
        series_meta_df = pd.read_csv(f"{self.config.BASE_PATH}/train_series_meta.csv")
        dataframe = pd.merge(train_df, series_meta_df, on="patient_id")
        dataframe["image_path"] = f"/content/drive/MyDrive/png_jjw" + "/" + dataframe.patient_id.astype(str) + "/" + dataframe.series_id.astype(str) + "/" + "img_256x256_d1_original"
        return dataframe

    @staticmethod
    def _assign_value(row, healthy_col, low_col=None, high_col=None):
        if row[healthy_col] == 1:
            return 0
        if low_col and row[low_col] == 1:
            return 1
        if high_col and row[high_col] == 1:
            return 2
        return None

    def process(self):
        self.df['bowel'] = self.df.apply(self._assign_value, healthy_col='bowel_healthy', axis=1)
        self.df['extravasation'] = self.df.apply(self._assign_value, healthy_col='extravasation_healthy', axis=1)
        self.df['kidney'] = self.df.apply(self._assign_value, healthy_col='kidney_healthy', low_col='kidney_low', high_col='kidney_high', axis=1)
        self.df['liver'] = self.df.apply(self._assign_value, healthy_col='liver_healthy', low_col='liver_low', high_col='liver_high', axis=1)
        self.df['spleen'] = self.df.apply(self._assign_value, healthy_col='spleen_healthy', low_col='spleen_low', high_col='spleen_high', axis=1)

    def _balance_data(self, col):
        negative = self.df[self.df[col] == 0]
        positive = self.df[self.df[col] == 1]
        num_samples = min(len(negative), len(positive))
        negative_samples = negative.sample(n=num_samples, random_state=self.config.SEED)
        positive_samples = positive.sample(n=num_samples, random_state=self.config.SEED)
        return pd.concat([negative_samples, positive_samples], axis=0)

    def get_balanced_data(self):
        bowel_dataframe = self._balance_data('bowel')
        extra_dataframe = self._balance_data('extravasation')
        kidney_dataframe = self._balance_data('kidney')
        liver_dataframe = self._balance_data('liver')
        spleen_dataframe = self._balance_data('spleen')
        return bowel_dataframe, extra_dataframe, kidney_dataframe, liver_dataframe, spleen_dataframe

    def split_data(self, dataframe):
        train_data, val_data = pd.DataFrame(), pd.DataFrame()

        for _, group in dataframe.groupby(self.config.TARGET_COLS):
            train_group, val_group = self._split_group(group)
            train_data = pd.concat([train_data, train_group], ignore_index=True)
            val_data = pd.concat([val_data, val_group], ignore_index=True)

        return train_data, val_data

    # @staticmethod
    def _split_group(self,group, test_size=0.2):
        if len(group) == 1:
            return (group, pd.DataFrame()) if np.random.rand() < test_size else (pd.DataFrame(), group)
        else:
            return train_test_split(group, stratify=group[self.config.TARGET_COLS], test_size=test_size, random_state=42)

class ImageProcessing:
    def __init__(self, config: Config):
        self.config = config
        self.random_flip_layer = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
        self.random_rotation_layer = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
        self.augmenter = self._initialize_augmenter()

    @staticmethod
    def _initialize_augmenter():
        cutout_params = {"height_factor": 0.2, "width_factor": 0.2}
        return CustomAugmenter(cutout_params)

    def decode_image_and_label(self, image_path, label):
        file_bytes1 = tf.io.read_file(image_path+'/image_001.png')
        image1 = tf.io.decode_png(file_bytes1, channels=1, dtype=tf.uint8)
        file_bytes2 = tf.io.read_file(image_path+'/image_002.png')
        image2 = tf.io.decode_png(file_bytes2, channels=1, dtype=tf.uint8)
        file_bytes3 = tf.io.read_file(image_path+'/image_003.png')
        image3 = tf.io.decode_png(file_bytes3, channels=1, dtype=tf.uint8)
        image = tf.concat([image1, image2, image3], axis=2)

        image = tf.image.resize(image, self.config.IMAGE_SIZE, method="bilinear")
        image = tf.cast(image, tf.float32) / 255.0

        label = tf.cast(label, tf.float32)

        return (image, label)

    def apply_augmentation(self, images, labels):
        augmented_images = self.augmenter(images, training=True)
        return (augmented_images, labels)

    def build_dataset(self, image_paths, labels):
        ds = (
            tf.data.Dataset.from_tensor_slices((image_paths, labels))
            .map(self.decode_image_and_label, num_parallel_calls=self.config.AUTOTUNE)
            .shuffle(self.config.BATCH_SIZE * 10)
            .batch(self.config.BATCH_SIZE)
            .map(self.apply_augmentation, num_parallel_calls=self.config.AUTOTUNE)
            .prefetch(self.config.AUTOTUNE)
        )
        return ds
    
class CustomAugmenter(tf.keras.layers.Layer):
    def __init__(self, cutout_params, **kwargs):
        super(CustomAugmenter, self).__init__(**kwargs)
        self.cutout_layer = keras_cv.layers.Augmenter([keras_cv.layers.RandomCutout(**cutout_params)])

    def call(self, inputs, training=None):
        if training:
            inputs = self.random_flip_layer(inputs)
            inputs = self.random_rotation_layer(inputs)
            inputs = self.cutout_layer(inputs)
        return inputs

def sensitivity(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    true_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

def build_binary_classification_model(warmup_steps, decay_steps, head_name):
    config=Config()
    inputs = keras.Input(shape=config.IMAGE_SIZE + [3,], batch_size=config.BATCH_SIZE)

    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_b3")
    backbone.include_rescaling = False
    x = backbone(inputs)

    gap = keras.layers.GlobalAveragePooling2D()
    x = gap(x)

    x_head = keras.layers.Dense(32, activation='silu')(x)

    output = keras.layers.Dense(1, name=head_name, activation='sigmoid')(x_head)

    print(f"[INFO] Building the {head_name} model...")
    model = keras.Model(inputs=inputs, outputs=output)

    cosine_decay = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=decay_steps,
        alpha=0.0,
        warmup_target=1e-3,
        warmup_steps=warmup_steps,
    )

    optimizer = keras.optimizers.Adam(learning_rate=cosine_decay)
    loss = keras.losses.BinaryCrossentropy()
    metrics = ["accuracy", sensitivity, specificity]

    print(f"[INFO] Compiling the {head_name} model...")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


def build_tertiary_classification_model(warmup_steps, decay_steps, head_name):
    config=Config()
    inputs = keras.Input(shape=config.IMAGE_SIZE + [3,], batch_size=config.BATCH_SIZE)

    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_b3")
    backbone.include_rescaling = False
    x = backbone(inputs)

    gap = keras.layers.GlobalAveragePooling2D()
    x = gap(x)

    x_head = keras.layers.Dense(32, activation='silu')(x)

    output = keras.layers.Dense(3, name=head_name, activation='softmax')(x_head)

    print(f"[INFO] Building the {head_name} model...")
    model = keras.Model(inputs=inputs, outputs=output)

    cosine_decay = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=decay_steps,
        alpha=0.0,
        warmup_target=1e-3,
        warmup_steps=warmup_steps,
    )

    optimizer = keras.optimizers.Adam(learning_rate=cosine_decay)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = ["accuracy"]

    print(f"[INFO] Compiling the {head_name} model...")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model

def train_and_save_model(config, img_processor, organ, dataframe):
    train_data, val_data = DataPreparation(config).split_data(dataframe)
    
    train_ds = img_processor.build_dataset(image_paths=train_data.image_path.tolist(), labels=train_data[organ].values)
    val_ds = img_processor.build_dataset(image_paths=val_data.image_path.tolist(), labels=val_data[organ].values)
    
    total_train_steps = train_ds.cardinality().numpy() * config.BATCH_SIZE * config.EPOCHS
    warmup_steps = int(total_train_steps * 0.10)
    decay_steps = total_train_steps - warmup_steps

    if organ in ["bowel", "extravasation"]:
        model = build_binary_classification_model(config, warmup_steps, decay_steps, organ)
    else:
        model = build_tertiary_classification_model(config, warmup_steps, decay_steps, organ)
    
    history = model.fit(train_ds, epochs=config.EPOCHS, validation_data=val_ds)

    model_filename = f"EfficinetnetB3_{organ}.keras"
    model_path = os.path.join(config.SAVE_DIR, model_filename)
    model.save(model_path)

    return history

class ImageInferenceProcessor:

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def _decode_image_python(image_path):
        image_path = image_path.numpy().decode('utf-8')

        dcm_files = [f for f in os.listdir(image_path)]
        dcm_files = sorted(dcm_files, key=lambda x: int(x.split('.')[0]))

        image_path_ = os.path.join(image_path, dcm_files[len(dcm_files)//2])
        image = pydicom.dcmread(image_path_)

        resized_image = cv2.resize(image.pixel_array, (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), interpolation=cv2.INTER_AREA).astype(np.uint8)

        images = np.zeros((Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), dtype=np.uint8)
        images[..., 0] = resized_image
        images[..., 1] = resized_image
        images[..., 2] = resized_image

        return tf.cast(images, tf.float32) / 255.0

    def decode_image(self, image_path):
        return tf.py_function(self._decode_image_python, [image_path], Tout=tf.float32)

    def build_inference_dataset(self, image_paths):
        ds = (
            tf.data.Dataset.from_tensor_slices(image_paths)
            .map(self.decode_image, num_parallel_calls=self.config.AUTOTUNE)
            .batch(self.config.BATCH_SIZE)
            .prefetch(self.config.AUTOTUNE)
        )
        return ds

def main():
    config = Config()
    data_prep = DataPreparation(config)
    data_prep.process()
    organ_dataframes = data_prep.get_balanced_data()

    img_processor = ImageProcessing(config)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()

    for organ, dataframe in zip(config.TARGET_COLS, organ_dataframes):
        history = train_and_save_model(config, img_processor, organ, dataframe)

        axes[0].plot(history.history['accuracy'], label='Training ' + organ)
        axes[1].plot(history.history['val_accuracy'], label='Validation ' + organ)

    axes[0].set_title("Training Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[0].set_xlabel('Epoch')
    axes[1].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def infer_main():
    config = Config()
    data_prep = DataPreparation(config)
    data_prep.process()
    organ_dataframes = data_prep.get_balanced_data()

    img_processor = ImageProcessing(config)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()

    meta_df = pd.read_csv(f"/content/drive/MyDrive/rsna_data/test_series_meta.csv")
    infer_processor = ImageInferenceProcessor(config)
    image_paths_for_inference = [path for path in meta_df.image_path.tolist() if os.path.exists(path)]
    inference_ds = infer_processor.build_inference_dataset(image_paths_for_inference)

    patient_ids = meta_df["patient_id"].unique()
    final_df = pd.DataFrame({'patient_id':patient_ids})
    BASE_PATH = "/content/drive/MyDrive/rsna_data"
    model_names = ["bowel", "extravasation", "kidney", "liver", "spleen"]
    for i, name in enumerate(model_names):
        model_filename = f"EfficinetnetB3_{name}.keras"
        model_path = os.path.join(BASE_PATH, "checkpoint", model_filename)
        if name =="bowel" or name=="extravasation":
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[sensitivity, specificity])
        else:
            model = tf.keras.models.load_model(model_path)
        
        model.summary()
        print(f"Output shape for model {name}: {model.output_shape}")

        # Moved the prediction logic inside the loop for each model
        patient_preds = np.zeros(shape=(len(patient_ids), model.output_shape[-1]), dtype="float32")

        for pidx, patient_id in tqdm(enumerate(patient_ids), total=len(patient_ids), desc="Patients "):
            print(f"Patient ID: {patient_id}")

            patient_df = meta_df[meta_df["patient_id"] == patient_id]
            patient_paths = [path for path in patient_df.image_path.tolist() if os.path.exists(path)]
            dtest = img_processor.build_dataset(patient_paths)
            pred = model.predict(dtest)

            print(f"Predictions shape for patient {patient_id}: {pred.shape}")

            if len(pred.shape) == 1:
                pred = pred.reshape(1, 1)
            if len(pred.shape) == 2:
                dim = pred.shape[1]
            else:
                raise ValueError(f"Unexpected shape for pred: {pred.shape}")

            pred = np.mean(pred.reshape(1, len(patient_paths), dim), axis=0)
            pred = np.max(pred, axis=0, keepdims=True)
            patient_preds[pidx, :model.output_shape[-1]] = pred.squeeze()

            del patient_df, patient_paths, dtest, pred; gc.collect()

        temp_df = pd.DataFrame(patient_preds, columns=[f"{name}_pred_{j}" for j in range(model.output_shape[-1])])
        final_df = pd.concat([final_df, temp_df], axis=1)

    return final_df
if __name__ == "__main__":
    main()