import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load your saved model
model = load_model('/home/uday/Documents/projects/deployment/dog_or_cat_classification_deployment/dog_or_cat_classification.keras')




# Prediction function
def predict_dog_or_cat(img):
    try:
        print("Type of img:", type(img))  # This will show you the type
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)

        if model.output_shape[-1] == 1:
            result = "Dog" if prediction[0][0] >= 0.5 else "Cat"
        else:
            result = "Dog" if np.argmax(prediction) == 1 else "Cat"

        return f"that's a {result}"
    except Exception as e:
        print(f"Error: {e}")
        return str(e)

  # This will make the error message show up in the UI


iface = gr.Interface(fn=predict_dog_or_cat, inputs=gr.Image(type="pil", image_mode="RGB"), outputs=gr.Textbox(),
                    title='Cat or Dog Classification App', description='Upload a Cat or Dog image.', theme='soft')
iface.launch()
