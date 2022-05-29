import io
from PIL import Image
import streamlit as st
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing import image



import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('my_model.h5')
class_names = ['degods', 'famous_fox_federation', 'solana_monkey_bussiness','others']

batch_size = 32
img_height = 180
img_width = 180


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        #return Image.open(BytesIO(image_data))
        return Image.open(io.BytesIO(image_data))


    else:
        return None



# img = Image.open(io.BytesIO(img_bytes))
# img = img.convert('RGB')
# img = img.resize(target_size=(img_height, img_width) , Image.NEAREST)
# img = image.img_to_array(img)


def predict(model,image_data,class_names):

    print(type(image_data))

    # img = tf.keras.utils.load_img(
    #     image_data, target_size=(img_height, img_width)
    # )

    #img = Image.open(io.BytesIO(img_bytes))

    batch_size = 32
    img_height = 180
    img_width = 180

    img = image_data.convert('RGB')
    #img = tf.image.resize(image_data,[img_height,img_width], method='nearest')
    img = tf.keras.preprocessing.image.smart_resize(img,[img_height,img_width],interpolation='nearest')
    img = image.img_to_array(img)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_predict = class_names[np.argmax(score)]
    score_predict = 100 * np.max(score)
    #score_predict = "{:.2f}".format(float(100 * np.max(score)))
    # "{:.3f}".format(float(prediction*100))
    #print(type(score_predict))


    return class_predict, score_predict

def main():
    st.title('NFT Image Classification')
    st.markdown('_created by Akmal Sabri_')
    model = tf.keras.models.load_model('my_model.h5')
    class_names = ['degods', 'famous_fox_federation', 'solana_monkey_bussiness','others']



    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        class_pred, score_pred = predict(model,image,class_names)

        if score_pred >= 99.99 :

            st.write(class_pred)
            st.write('**Probability: **',score_pred,'%')

        else :

            st.write("Not sure")





if __name__ == '__main__':
    main()
