#20-2-25

# Sample dataset: List of messages with labels
data = [
    {"message": "Congratulations! You won a free lottery ticket!", "label": "spam"},
    {"message": "Hey, are we still meeting tomorrow?", "label": "ham"},
    {"message": "Win cash prizes now!!! Click here.", "label": "spam"},
    {"message": "Can you send me the report by evening?", "label": "ham"},
    {"message": "Get free coupons by signing up today!", "label": "spam"},
    {"message": "Lunch at 1 pm?", "label": "ham"},
]

# List of spam keywords
spam_keywords = ["free", "win", "cash", "click", "prize", "lottery"]

# Function to classify messages
def classify_message(message):
    message = message.lower()  # Convert to lowercase for uniformity
    for keyword in spam_keywords:
        if keyword in message:
            return "spam"
    return "ham"

# Test the classifier on the dataset
correct_predictions = 0

for item in data:
    prediction = classify_message(item["message"])
    print(f"Message: '{item['message']}' | Actual: {item['label']} | Predicted: {prediction}")
    if prediction == item["label"]: #if the predection matched the original label of data item
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
