import streamlit as st
import os, torch, shutil, time, pandas as pd
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

start_time_load_model = time.time()
model_checkpoint_test = "microsoft/beit-base-patch16-384"
inf_image_processor = AutoImageProcessor.from_pretrained(model_checkpoint_test)
model_test = AutoModelForImageClassification.from_pretrained("../Model/beit-base-patch16-384-21L_15E_8B_5e-05_0.2")
end_time_load_model = time.time()
print(f"\nDuration to load model: {end_time_load_model-start_time_load_model} seconds")

# Function to perform predictions
def predict(img):
    image = Image.open(img)
    encoding = inf_image_processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model_test(**encoding)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model_test.config.id2label[predicted_class_idx]

def main():
    st.title('Bone Marrow Cell Classification')
    all_pred = {}       

    # Setup button and variables states
    if 'predict_button_pressed' not in st.session_state:
        st.session_state.predict_button_pressed = False
    if 'all_pred' in st.session_state:
        all_pred = st.session_state.all_pred
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    # Text input
    # folder_path = st.sidebar.text_input('Enter path to folder:', " ")

    # Allow user to select a folder containing images
    uploaded_file = st.sidebar.file_uploader("Upload Images", type=([".jpg", ".png"]), accept_multiple_files=True, key = f"uploader_{st.session_state.uploader_key}")

    # Predict Button
    if st.sidebar.button('Predict'):
        start_time_predict = time.time()
        print("Predict Button Pressed")
        # If there are files uploaded
        if len(uploaded_file) != 0:
            #If Predict button has been pressed before, reset saved predictions
            if st.session_state.predict_button_pressed == True:
                all_pred = {}
            #image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('jpg', 'png'))]

            # Get all uploaded images and run prediction
            for img in uploaded_file:
                prediction = predict(img)
                if prediction in all_pred:
                    # Existing Prediction
                    all_pred[prediction]["img_paths"].append(img)
                    all_pred[prediction]["count"] = all_pred[prediction]["count"] + 1
                else:
                    # New Prediction
                    all_pred.update({prediction: {"count": 1, "img_paths": [img]}})
            st.session_state.predict_button_pressed = True
            st.session_state.all_pred = all_pred
            end_time_predict = time.time()
            print(f"Prediction Duration: {end_time_predict - start_time_predict} seconds")
        else:
            st.write("Please Upload Files.")

    # Clear Button
    if st.sidebar.button('Clear'):
        print("Clear Button Pressed")
        st.session_state.predict_button_pressed = False
        st.session_state.all_pred = {}
        st.session_state.uploader_key += 1 
        st.rerun()

    if st.session_state.predict_button_pressed == True:
        # Display total count of each category
        table_data = []
        total = 0
        for name in all_pred:
            table_data.append([name, all_pred[name]["count"]])
            total += all_pred[name]["count"]
        table_df = pd.DataFrame(table_data, columns=['Cell Name', 'Count'])
        st.sidebar.write(f"Total images predicted: {total}")
        st.sidebar.table(table_df.set_index('Cell Name'))

        # Dropdown box to choose category
        option = st.selectbox(
                'Choose:',
                list(all_pred.keys()),
                index = 0
            )
        
        # Display all images under that category
        for image in all_pred[option]["img_paths"]:
            img_data = Image.open(image)
            st.image(img_data, caption=image.name, width= img_data.width)   
            
if __name__ == '__main__':
    main()
