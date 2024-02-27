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

from data_processing.load_dataset import load_dataset
from data_processing.data_augmentation import balance_genders, augment_data
from sklearn.model_selection import train_test_split


V1_MODEL_TRAINING_HISTORY_PATH = "cnn/model-versions/v1/model-training-history.csv"
V1_MODEL_PATH = "cnn/model-versions/v1/"


df = load_dataset()
df = balance_genders(df)

X = df
y = df['Age']

# Split data into training (80%) , validation (10%) and test sets (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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
