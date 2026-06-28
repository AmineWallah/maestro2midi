import tensorflow as tf
from model import build_unet
from data_pipeline import MAESTROGenerator

def main():
    train_generator = MAESTROGenerator('maestro_chunks.h5', 'train', batch_size=32)
    validation_generator = MAESTROGenerator('maestro_chunks.h5', 'validation', batch_size=32)
    model = build_unet()

    model.compile(
        optimizer='adam',
        loss={
            'frame_output': 'binary_crossentropy',
            'onset_output': 'binary_crossentropy',
        },
        loss_weights={
            'frame_output': 1.0,
            'onset_output': 1.0,  # bump this up (e.g. 2.0) if onsets undertrains
        },
        metrics={
            'frame_output': [tf.keras.metrics.Precision(name='precision'),
                             tf.keras.metrics.Recall(name='recall')],
            'onset_output': [tf.keras.metrics.Precision(name='precision'),
                             tf.keras.metrics.Recall(name='recall')],
        },
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ]
    )

    model.save('final_model.keras')


if __name__ == "__main__":
    main()


