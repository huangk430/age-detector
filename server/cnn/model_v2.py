import numpy as np
import pandas as pd

from keras.layers import (
    Input, 
    SeparableConv2D, 
    Activation, 
    MaxPooling2D, 
    BatchNormalization,
    SpatialDropout2D,
    Flatten,
    Dense,
    Dropout,
)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, EarlyStopping

from data_processing.load_dataset import load_dataset
from data_processing.data_augmentation import balance_genders, augment_data
from sklearn.model_selection import train_test_split


V2_MODEL_TRAINING_HISTORY_PATH = "server/cnn/model_versions/v2/model-training-history.csv"
V2_MODEL_PATH = "server/cnn/model_versions/v2/age-i-model-v2.h5"


def train(X_train, X_val):
    train_generator, val_generator = augment_data(train_df=X_train, val_df=X_val)

    input_shape = (180, 180, 3)
    inputs = Input(shape=input_shape)

    x = SeparableConv2D(64, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(256, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SpatialDropout2D(0.15)(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(512, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = SeparableConv2D(512, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    x = Dropout(0.30)(x)

    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = Dropout(0.30)(x)
    x = BatchNormalization()(x)

    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = Dropout(0.30)(x)
    x = BatchNormalization()(x)

    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = Dropout(0.30)(x)
    x = BatchNormalization()(x)

    x = Dense(64)(x)
    x = Activation("relu")(x)
    x = Dropout(0.30)(x)
    x = BatchNormalization()(x)

    x = Dense(1)(x)
    x = Activation("relu")(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(
        loss=["mse"],
        optimizer="adam",
        metrics=["mae", "mse"]
    )

    training_step_size = train_generator.n / train_generator.batch_size
    validation_step_size = val_generator.n / val_generator.batch_size

    csv_logger = CSVLogger(V2_MODEL_TRAINING_HISTORY_PATH)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    training_history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=training_step_size, 
        validation_data=val_generator,
        validation_steps=validation_step_size, 
        epochs=50,
        callbacks=[csv_logger, early_stopping]
    )

    with open('server/cnn/model_versions/v2/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    print(training_history)

    model.save(V2_MODEL_PATH)


def test_model(X_test):
    batch_size = 64
    
    model = load_model(V2_MODEL_PATH)
    test_datagen = ImageDataGenerator(rescale=1./255.) 

    steps = len(X_test) // batch_size
    sample_size = batch_size * steps
    
    X_test = X_test[:sample_size]

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=X_test,
        directory="server/dataset/wiki_crop",
        x_col="Image Path",
        y_col="Age",
        batch_size=batch_size,
        shuffle=False,
        class_mode="raw",
        target_size=(180,180)
    )

    # Make predictions batch by batch
    predictions = []
    for i in range(steps):
        batch = next(test_generator)
        batch_images = batch[0]  # batch: Tuple(image, age)
        batch_predictions = model.predict(batch_images)
        
        # Print each prediction along with its index and true target
        for j, (prediction, true_target) in enumerate(zip(batch_predictions, batch[1])):
            print(f"Index: {i * test_generator.batch_size + j}, Prediction: {prediction}, True Target: {true_target}")
        
        predictions.append(batch_predictions)

    predictions = np.concatenate(predictions)

    true_targets = X_test["Age"].values  

    results_df = pd.DataFrame({
        'Index': X_test.index,
        'Predictions': predictions.flatten(),
        'True Targets': true_targets
    })


    # Save the results to a CSV file
    results_df.to_csv('server/cnn/model_versions/v2/test_results.csv', index=False)


df = load_dataset()
df = balance_genders(df)

X = df
y = df['Age']

# Split data into training (80%) , validation (10%) and test sets (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train(X_train, X_val)
test_model(X_test)
