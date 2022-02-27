# %%
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import tensorflowjs as tfjs
from pathlib import Path
from sklearn.utils import class_weight
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics, model_selection, preprocessing
from sklearn.model_selection import StratifiedKFold


# %%
image_size = (64, 64)
batch_size = 32
subset = None
main_directory = Path("/media/hdd/github/emotion_recog_js/web/fer2013/")
# keras.mixed_precision.set_global_policy("mixed_float16")
# %%
print(tf.config.list_physical_devices('GPU'))
# %%
sub_dir = os.listdir(main_directory/'train')
sb = {x: len(os.listdir(Path(main_directory/'train'/x))) for x in sub_dir}
sb
# %%
all_ims = glob.glob(str(main_directory) + "/*/*/*.png")
all_ims[0]

len(all_ims)


def create_label(x):
    return x.split("/")[-2]


df = pd.DataFrame.from_dict(
    {x: create_label(x) for x in all_ims}, orient="index"
).reset_index()
df.columns = ["image_id", "label"]

df.head()

df.label.nunique()

df.label.value_counts()
#%%
def oversample(df):
    classes = df.label.value_counts().to_dict()
    most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df['label'] == key]) 
    classes_sample = []
    for i in range(1,len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe,classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df
#%%
df = oversample(df)
#%%
df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
stratify = StratifiedKFold(n_splits=5)
for i, (t_idx, v_idx) in enumerate(
    stratify.split(X=df.image_id.values, y=df.label.values)
):
    df.loc[v_idx, "kfold"] = i
    df.to_csv("train_folds.csv", index=False)
# %%
df = pd.read_csv("train_folds.csv")
df.head(10)
# %%
train = df.loc[df["kfold"] != 1]
val = df.loc[df["kfold"] == 1]

if subset != None:
    train = df.loc[df["kfold"] != 1].head(subset)
    val = df.loc[df["kfold"] == 1].head(subset)
# %%
TRAIN_DATAGEN = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True)

TEST_DATAGEN = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.)

train_ds = TRAIN_DATAGEN.flow_from_dataframe(dataframe=train, directory=main_directory/'train',
                                             x_col="image_id", y_col="label",
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             target_size=image_size,
                                             subset='training',
                                             )

val_ds = TEST_DATAGEN.flow_from_dataframe(dataframe=val, directory=main_directory/'test',

                                          x_col="image_id", y_col="label",
                                          batch_size=batch_size,
                                          target_size=image_size,
                                          class_mode='categorical',)


# %%
encoded_labels = os.listdir(main_directory/'train')
encoded_labels
# %%
# %%
data_augmentation = keras.Sequential([
    # layers.RandomRotation(0.2),
    # layers.RandomFlip(),
    # layers.experimental.preprocessing.RandomContrast(.2)
]
)
# %%


def build_net(n_class):
    net = keras.Sequential()

    net.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            input_shape=(64, 64, 3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_1'
        )
    )
    net.add(layers.BatchNormalization(name='batchnorm_1'))
    net.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )
    )
    net.add(layers.BatchNormalization(name='batchnorm_2'))

    net.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
    net.add(layers.Dropout(0.4, name='dropout_1'))

    net.add(
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )
    )
    net.add(layers.BatchNormalization(name='batchnorm_3'))
    net.add(
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )
    )
    net.add(layers.BatchNormalization(name='batchnorm_4'))

    net.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
    net.add(layers.Dropout(0.4, name='dropout_2'))

    net.add(
        layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        )
    )
    net.add(layers.BatchNormalization(name='batchnorm_5'))
    net.add(
        layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        )
    )
    net.add(layers.BatchNormalization(name='batchnorm_6'))

    net.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'))
    net.add(layers.Dropout(0.5, name='dropout_3'))

    net.add(layers.Flatten(name='flatten'))

    net.add(
        layers.Dense(
            128,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense_1'
        )
    )
    net.add(layers.BatchNormalization(name='batchnorm_7'))

    net.add(layers.Dropout(0.6, name='dropout_4'))

    net.add(
        layers.Dense(
            n_class,
            activation='softmax',
            name='out_layer'
        )
    )

    return net


# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model = build_net(len(encoded_labels))
# %%
# class_weights = class_weight.compute_class_weight(
#     'balanced',
#     classes=np.unique(train_ds.classes),
#     y=train_ds.classes)
# train_class_weights = dict(enumerate(class_weights))

# %%
keras.utils.plot_model(model, show_shapes=True)
# %%

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    keras.callbacks.ModelCheckpoint("./logs/save_at_{epoch}.h5"),
    keras.callbacks.ProgbarLogger(count_mode="samples", stateful_metrics=None),
    lr_scheduler,
    early_stopping
]
loss_fn = keras.losses.CategoricalCrossentropy()
opt = keras.optimizers.Adam(1e-3)


model.compile(
    optimizer=opt,
    loss=loss_fn,
    metrics=["accuracy"],
)
#%%

# model.load("./web/effnet")
# %%
epochs = 10
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
    # class_weight=train_class_weights
)
# %%
model.save("./web/effnet")
# %%
prediction_classes = np.array([])
true_classes = np.array([])

for x, y in val_ds:
    prediction_classes = np.concatenate([prediction_classes,
                                         np.argmax(model.predict(x), axis=-1)])
    true_classes = np.concatenate(
        [true_classes, np.argmax(y, axis=-1)])
# %%
dl = confusion_matrix(true_classes, prediction_classes)
df_cm = pd.DataFrame(dl, encoded_labels, encoded_labels)
# plt.figure(figsize=(10,7))
sns.set(font_scale=1.4)  # for label size
sns.heatmap(df_cm, annot=False, annot_kws={"size": 16})  # font size

plt.show()
# %%
import tensorflow_model_optimization as tfmot
model = tfmot.sparsity.keras.prune_low_magnitude(model)
tfjs.converters.save_keras_model(model, "./web/effnet")
# %%
converter = tf.lite.TFLiteConverter.from_saved_model("./web/effnet")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
# %%
with open("./web/working_quantized.tflite", "wb") as output_file:
    output_file.write(tflite_quant_model)

# %%
