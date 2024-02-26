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
    Model
)


def cnn():
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
        metrics=["accuracy"]
    )

    # plot_model(model)

    

