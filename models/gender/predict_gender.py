from age_and_gender import AgeAndGender

data = AgeAndGender()
data.load_shape_predictor('models/gender/pretrained/shape_predictor_5_face_landmarks.dat')
data.load_dnn_gender_classifier('models/gender/pretrained/dnn_gender_classifier_v1.dat')
data.load_dnn_age_predictor('models/gender/pretrained/dnn_age_predictor_v1.dat')


def get_gender(image):
    # image = Image.open('test-image.jpg').convert("RGB")
    result = data.predict(image)

    return result[0]['gender']