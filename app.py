import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from PIL import Image
import os

def main():
    st.title("Fall Detection System")
    
    dataset_path = st.text_input("Enter dataset path:", r"code\fall_dataset")

    # Ensure session state variables are initialized
    if "train_generator" not in st.session_state:
        st.session_state.train_generator = None
    if "val_generator" not in st.session_state:
        st.session_state.val_generator = None
    if "model" not in st.session_state:
        st.session_state.model = None

    if st.button("Load Dataset"):
        if os.path.exists(dataset_path):
            train_datagen = ImageDataGenerator(
                rescale=1.0/255.0,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )

            st.session_state.train_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='binary',
                subset='training'
            )

            st.session_state.val_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='binary',
                subset='validation'
            )

            st.success("Dataset Loaded Successfully!")
        else:
            st.error("Dataset path is incorrect or missing!")

    if st.button("Train Model"):
        if st.session_state.train_generator is None or st.session_state.val_generator is None:
            st.error("Please load the dataset before training!")
        else:
            base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            base_model.trainable = False

            model = Sequential([
                base_model,
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            st.text("Model Summary")
            st.text(model.summary())

            history = model.fit(
                st.session_state.train_generator,
                validation_data=st.session_state.val_generator,
                epochs=10,
                verbose=1
            )
            st.success("Model Training Completed!")
            
            # Plot training history
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(history.history['accuracy'], label='Train Accuracy')
            ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax[0].legend()
            ax[0].set_title('Model Accuracy')
            
            ax[1].plot(history.history['loss'], label='Train Loss')
            ax[1].plot(history.history['val_loss'], label='Validation Loss')
            ax[1].legend()
            ax[1].set_title('Model Loss')
            
            st.pyplot(fig)

            st.session_state.model = model

    uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict Fall") and st.session_state.model:
            model = st.session_state.model
            img = img.resize((224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)[0][0]
            label = "No Fall Detected ✅" if prediction > 0.5 else "Fall Detected! 🚨"
            color = "green" if prediction > 0.5 else "red"
            
            st.markdown(f"<h3 style='color:{color};'>{label}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
