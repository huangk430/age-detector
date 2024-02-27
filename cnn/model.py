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

    # plot_model(model)

    STEP_SIZE_TRAIN=15536/64 # TODO: replace these with variables later
    STEP_SIZE_VALID=1942/64

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN, 
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VALID, 
        epochs=45
    )

    with open("model-output.txt") as f:
        f.write(history)
