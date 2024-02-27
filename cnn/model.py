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
from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping


V1_MODEL_TRAINING_HISTORY_PATH = "cnn/model-versions/v1/model-training-history.csv"
V1_MODEL_PATH = "cnn/model-versions/v1/"


def cnn(train_generator, val_generator):
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

    x = Flatten()(x)

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

    csv_logger = CSVLogger(V1_MODEL_TRAINING_HISTORY_PATH)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
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

    print(training_history)

    model.save(V1_MODEL_PATH)
