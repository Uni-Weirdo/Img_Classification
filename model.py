import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Define a list of accepted keywords for food and fitness-related labels
food_fitness_keywords = [
    "apple", "banana", "burger", "cake", "cheese", "chicken", "coffee", "egg", "fruit",
    "gym", "barbell", "dumbbell", "exercise", "fitness", "treadmill", "weight", "yoga", 
    "pizza", "salad", "sushi", "steak", "pasta", "chocolate", "donut", "orange", "strawberry",
    "panner", "roti", "dosa", "idli"
]

def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = img_to_array(img)  # Convert image to array
    img_array = preprocess_input(img_array)  # Preprocess the image
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)  # Get top 3 predictions

    # Print predictions for debugging
    print("Predictions:", decoded_predictions)

    # Extract top predicted label
    top_label = decoded_predictions[0][0][1].lower()  # Get the name of the predicted class

    # Check if the predicted label matches food/fitness related terms
    for keyword in food_fitness_keywords:
        if keyword in top_label:
            return "Accept (Food/Fitness)"

    # If no match found, reject
    return "Reject (Not Food/Fitness)"

