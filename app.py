from flask import Flask, render_template, request
import random
import json
import torch
from model import NeuralNet
from nltk_util import bag_of_words, tokenize
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

app = Flask(__name__)


@app.route("/")
def home():
    reply = ''
    return render_template("index.html", reply=reply)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    sentence = request.form["user_message"]

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]


    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                if tag == 'datetime':
                    reply = datetime.datetime.now()
                    return render_template("index.html", reply=reply)
                else:
                    reply = random.choice(intent['responses'])
                    return render_template("index.html", reply=reply)


    else:
        return render_template("index.html", reply="I do not understand")


if __name__ == "__main__":
    app.run(debug=True)

"""
import random
import json
import torch
from model import NeuralNet
from nltk_util import bag_of_words, tokenize
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Hussain"
print("Let's chat! type 'quit' to exit")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                if tag == 'datetime':
                    print(f"{bot_name}: {datetime.datetime.now()}")
                else:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")


    else:
        print(f"{bot_name}: I do not understand...")

"""
