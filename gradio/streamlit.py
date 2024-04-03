import streamlit as st
import os, torch, shutil
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

model_checkpoint_test = "google/vit-base-patch16-224"
inf_image_processor = AutoImageProcessor.from_pretrained(model_checkpoint_test)
model_test = AutoModelForImageClassification.from_pretrained("/home/dxd_jy/joel/Capstone/Model/best-vit-base-patch16-224-10L_20E_8B_5e-05_0.3")

# Function to make predictions
def predict(img_path):
    image = Image.open(img_path)
    encoding = inf_image_processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model_test(**encoding)
        logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model_test.config.id2label[predicted_class_idx]


def main():
    st.title('Bone Marrow Cell Classification')
    all_pred = {}

    if 'button_pressed' not in st.session_state:
        st.session_state.button_pressed = False
    
    if 'all_pred' in st.session_state:
        all_pred = st.session_state.all_pred

    # Allow user to select a folder containing images
    folder_path = st.sidebar.text_input('Enter path to folder:', "/home/dxd_jy/joel/Capstone/Training_Testing/Test/Test_Images/Images")
    #folder_path = st.file_uploader("Select Folder", type="directory", accept_multiple_files=False)

    if st.sidebar.button('Predict'):
        if st.session_state.button_pressed == True:
            all_pred = {}
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('jpg', 'png'))]
        for img_path in image_files:
            prediction = predict(img_path)
            if prediction in all_pred:
                all_pred[prediction]["img_paths"].append(img_path)
                all_pred[prediction]["count"] = all_pred[prediction]["count"] + 1
            else:
                all_pred.update({prediction: {"count": 1, "img_paths": [img_path]}})
        st.session_state.button_pressed = True
        st.session_state.all_pred = all_pred
    
    if st.session_state.button_pressed == True:
        for name in all_pred:
            st.sidebar.write(name)
        option = st.selectbox(
                'Choose:',
                list(all_pred.keys()),
                index = 0
            )

        for image in all_pred[option]["img_paths"]:
            img_data = Image.open(image)
            file_name = os.path.basename(image)
            st.image(img_data, caption=file_name, width= img_data.width)   
            
if __name__ == '__main__':
    main()
