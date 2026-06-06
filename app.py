import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AnimalVision AI",
    page_icon="🐾",
    layout="centered"
)


# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #1e293b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 900;
    color: #38bdf8;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.hero-card {
    background: rgba(255, 255, 255, 0.08);
    padding: 25px;
    border-radius: 22px;
    box-shadow: 0px 10px 35px rgba(0, 0, 0, 0.4);
    margin-bottom: 25px;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.result-box {
    background: rgba(34, 197, 94, 0.13);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(34, 197, 94, 0.45);
    margin-top: 20px;
}

.prediction-title {
    font-size: 32px;
    font-weight: 800;
    color: #22c55e;
    margin-bottom: 8px;
}

.confidence-text {
    font-size: 20px;
    color: #e2e8f0;
}

.top-prediction-card {
    background: rgba(255, 255, 255, 0.07);
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 10px;
    border: 1px solid rgba(255, 255, 255, 0.10);
}

.custom-progress-bg {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    height: 14px;
    margin: 8px 0 18px 0;
    overflow: hidden;
}

.custom-progress-fill {
    background: linear-gradient(90deg, #22c55e, #38bdf8);
    height: 14px;
    border-radius: 12px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #22c55e, #38bdf8);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 15px 24px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 6px 22px rgba(56, 189, 248, 0.45);
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
}

[data-testid="stMarkdownContainer"] {
    color: white;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    font-size: 14px;
}

.warning-box {
    background: rgba(250, 204, 21, 0.12);
    padding: 16px;
    border-radius: 14px;
    border: 1px solid rgba(250, 204, 21, 0.45);
    margin-top: 15px;
    color: #fef9c3;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================
st.markdown("<h1 class='main-title'>🐾 AnimalVision AI</h1>", unsafe_allow_html=True)

st.markdown(
    "<p class='subtitle'>Custom animal image classifier trained on 90 animal classes</p>",
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero-card">
    <h3>Upload an animal image</h3>
    <p>
    This AI app uses a custom-trained MobileNetV2 transfer learning model.
    Upload a clear animal image and the model will predict the animal class
    with confidence scores.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# LOAD CLASS NAMES
# =====================================================
@st.cache_data
def load_class_names():
    with open("class_names.txt", "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names


# =====================================================
# BUILD MODEL ARCHITECTURE
# This avoids the old .h5 TrueDivide loading issue
# =====================================================
def build_model(num_classes):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Same preprocessing used during training
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model


@st.cache_resource
def load_custom_model():
    class_names = load_class_names()
    num_classes = len(class_names)

    model = build_model(num_classes)

    # animal_model.h5 must be in the same folder as app.py
    model.load_weights("animal_model.h5")

    return model


try:
    with st.spinner("Loading AI model..."):
        class_names = load_class_names()
        model = load_custom_model()

    st.success(f"Model loaded successfully! Total animal classes: {len(class_names)}")

except Exception as e:
    st.error("Model loading failed.")
    st.write("Make sure these files are uploaded in your Hugging Face Space:")
    st.code("animal_model.h5\nclass_names.txt\napp.py")
    st.write("Error details:")
    st.write(e)
    st.stop()


# =====================================================
# HTML PROGRESS BAR
# =====================================================
def html_progress(confidence):
    percent = float(confidence) * 100

    if percent < 0:
        percent = 0
    if percent > 100:
        percent = 100

    st.markdown(
        f"""
        <div class="custom-progress-bg">
            <div class="custom-progress-fill" style="width: {percent}%;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# ANIMAL FACTS
# =====================================================
animal_facts = {
    "antelope": "Antelopes are fast-running animals commonly found in grasslands.",
    "badger": "Badgers are strong diggers and usually live in underground burrows.",
    "bat": "Bats are the only mammals capable of true flight.",
    "bear": "Bears have an excellent sense of smell and are very powerful animals.",
    "bee": "Bees are important pollinators and help plants reproduce.",
    "beetle": "Beetles are one of the largest groups of insects in the world.",
    "bison": "Bison are large grazing animals with strong shoulders and thick fur.",
    "boar": "Boars are wild relatives of domestic pigs.",
    "butterfly": "Butterflies go through complete metamorphosis from caterpillar to adult.",
    "cat": "Cats have excellent night vision and strong jumping ability.",
    "caterpillar": "Caterpillars later transform into butterflies or moths.",
    "chimpanzee": "Chimpanzees are intelligent primates closely related to humans.",
    "cockroach": "Cockroaches are very adaptable insects.",
    "cow": "Cows are social animals and can recognize familiar faces.",
    "coyote": "Coyotes are clever wild canines.",
    "crab": "Crabs usually walk sideways and have strong claws.",
    "crow": "Crows are highly intelligent birds.",
    "deer": "Deer are graceful animals often found in forests and grasslands.",
    "dog": "Dogs are loyal animals with an excellent sense of smell.",
    "dolphin": "Dolphins are intelligent marine mammals.",
    "donkey": "Donkeys are strong working animals.",
    "dragonfly": "Dragonflies are fast-flying insects often found near water.",
    "duck": "Ducks are water birds with webbed feet.",
    "eagle": "Eagles are powerful birds of prey with excellent eyesight.",
    "elephant": "Elephants are highly intelligent animals with strong memory.",
    "flamingo": "Flamingos are known for their pink color and long legs.",
    "fox": "Foxes are clever animals known for adaptability.",
    "goat": "Goats are agile animals and can climb rough surfaces.",
    "goldfish": "Goldfish are freshwater fish often kept as pets.",
    "gorilla": "Gorillas are strong, intelligent primates.",
    "horse": "Horses can sleep both standing up and lying down.",
    "kangaroo": "Kangaroos use powerful back legs to jump long distances.",
    "koala": "Koalas mainly eat eucalyptus leaves.",
    "leopard": "Leopards are strong climbers and often rest in trees.",
    "lion": "Lions live in social groups called prides.",
    "monkey": "Monkeys are intelligent animals and communicate socially.",
    "panda": "Pandas spend much of their day eating bamboo.",
    "parrot": "Parrots are colorful birds that can mimic sounds.",
    "penguin": "Penguins are flightless birds adapted for swimming.",
    "rabbit": "Rabbits have strong hind legs and can run fast.",
    "rhinoceros": "Rhinoceroses are large animals known for their horns.",
    "shark": "Sharks are powerful fish and important ocean predators.",
    "snake": "Snakes are legless reptiles found in many environments.",
    "squirrel": "Squirrels are small rodents known for climbing and storing food.",
    "tiger": "Tigers are the largest wild cats and have unique stripe patterns.",
    "turtle": "Turtles are reptiles with protective shells.",
    "whale": "Whales are large marine mammals.",
    "wolf": "Wolves live and hunt in packs.",
    "zebra": "Every zebra has a unique stripe pattern."
}


def get_animal_fact(label):
    label = label.lower()

    for animal, fact in animal_facts.items():
        if animal in label:
            return fact

    return "This is an interesting animal. Try uploading a clearer image for better prediction."


# =====================================================
# IMAGE PREPROCESSING AND PREDICTION
# =====================================================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_animal(image):
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image, verbose=0)[0]

    top_indices = predictions.argsort()[-5:][::-1]

    results = []

    for index in top_indices:
        animal_name = class_names[index]
        confidence = float(predictions[index])
        results.append((animal_name, confidence))

    return results


# =====================================================
# MAIN UI
# =====================================================
st.markdown("### Upload Image")

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Classify Animal"):
        with st.spinner("Analyzing image..."):
            results = predict_animal(image)

        top_animal, top_confidence = results[0]

        clean_animal = top_animal.replace("_", " ").title()
        confidence_percent = top_confidence * 100

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        st.markdown(
            f"<p class='prediction-title'>Prediction: {clean_animal}</p>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p class='confidence-text'>Confidence: {confidence_percent:.2f}%</p>",
            unsafe_allow_html=True
        )

        html_progress(top_confidence)

        fact = get_animal_fact(top_animal)
        st.info(f"🐾 Animal Fact: {fact}")

        if top_confidence < 0.50:
            st.markdown(
                """
                <div class="warning-box">
                    ⚠️ Confidence is low. Try uploading a clearer image with one animal visible.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Top 5 Predictions")

        for i, (animal_name, confidence) in enumerate(results):
            clean_name = animal_name.replace("_", " ").title()
            confidence_percent = confidence * 100

            st.markdown(
                f"""
                <div class="top-prediction-card">
                    <b>{i + 1}. {clean_name}</b><br>
                    Confidence: {confidence_percent:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            html_progress(confidence)

else:
    st.warning("Please upload an animal image to start classification.")


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🐾 AnimalVision AI")
st.sidebar.write("Custom Animal Classifier")
st.sidebar.write(f"Total Classes: {len(class_names)}")

with st.sidebar.expander("Show animal classes"):
    for name in class_names:
        st.write(name.replace("_", " ").title())


# =====================================================
# FOOTER
# =====================================================
st.markdown(
    "<p class='footer'>Built with Python, Streamlit, TensorFlow, MobileNetV2, and Hugging Face Spaces</p>",
    unsafe_allow_html=True
)
