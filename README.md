
**QuesNet: Question Answering System**

QuesNet is a chatbot designed to provide answers to questions based on a provided corpus, which can be in PDF, Word, or Excel format. It utilizes a pre-trained BERT model fine-tuned on the SQuAD 2.0 dataset. If an answer is not found in the provided information, it gracefully informs the user.

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
