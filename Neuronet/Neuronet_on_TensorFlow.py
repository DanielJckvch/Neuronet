import numpy as np
import scipy
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
#Запрет на вывод предупреждений


batch_size = 32#Размер партии при обучении
img_height = 180#Высота, к которой приводятся картинки
img_width = 180#Ширина, к которой приводятся картинки
epochs = 15#Число эпох обучения

#TRAIN_DATA_PATH = 'D:\\Программы\\Нейросеть к учебной практике\\Animals-10 dataset\\raw-img'
#MODEL_STORE_PATH = 'D:\\Программы\\Нейросеть к учебной практике\\Model\\'
#IMAGE_PATH = 'D:\\Программы\\Нейросеть к учебной практике\\Neuronet\\Neuronet\\image.jpg'

#Выбор режима работы с сетью
have_model = int(input('Enter mode: 0-training, other-image checking: '))


if (have_model == 0):#Обучение
    #Выбор обучения с выводом графика или без
    to_plot = int(input('Show train plot? 0-no, other-yes: '))
    #Подготовка набора данных
    TRAIN_DATA_PATH = input('Enter train data path: ')
    train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DATA_PATH, validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DATA_PATH, validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    #Описание сети
    data_augmentation = Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    ])
    num_classes = len(class_names)
    model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])
    #Обучение сети и её сохранение
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(MODEL_STORE_PATH+'Net')
    if(to_plot!=0):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
    

else:#Проверка картинки
    #Загрузка картинки и сохранённой модели
    MODEL_STORE_PATH = input('Enter model store path: ')
    IMAGE_PATH = input('Enter path to the image with its filename: ')
    model = tf.keras.models.load_model(MODEL_STORE_PATH+'Net')
    img = tf.keras.utils.load_img(IMAGE_PATH, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    #Определение наличия кота на картинке
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    if (np.argmax(score) == 5):
        print("With probability {:.2f}% there is cat on the picture".format(100 * np.max(score)))
    else:
        print('There is no cat on the picture')
    #print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))

