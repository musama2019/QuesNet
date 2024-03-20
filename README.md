
#QuesNet

Introducing QuesNet, your intelligent companion for instant answers to all your queries within your organization's vast corpus of documents. QuesNet is not just any chatbot; it's your go-to solution for unlocking the wealth of knowledge buried within PDFs, Word documents, or Excel sheets. Powered by a sophisticated pre-trained BERT model, fine-tuned on the SQuAD 2.0 dataset, QuesNet delivers accurate responses with lightning speed. Its seamless integration with various file formats ensures effortless access to information. And here's the beauty of it – if QuesNet can't find an answer, it gracefully informs you, saving time and frustration. With QuesNet by your side, navigating through your organization's documentation has never been smoother, enabling you to make informed decisions with confidence.

**Key Components:**

**Training:** The training.py file trains the pre-trained BERT model on the SQuAD 2.0 dataset, enabling QuesNet to understand and respond to questions effectively.

**Evaluation:** The evaluate.py file evaluates the trained model, ensuring its accuracy and reliability in providing answers.

**Documentation:** A comprehensive report detailing the project's methodology, findings, and results is included in the repository, providing insights into the development process and outcomes.

**Frontend:** The Flask web application, containing index.html as the frontend and app.py as the backend, serves as the user interface for QuesNet. Users can input their questions and receive answers seamlessly.

**Usage:**

Install dependencies.
Train the model using training.py.
Evaluate the model's performance with evaluate.py.
Start the Flask app using python app.py and navigate to the provided URL to interact with QuesNet.
