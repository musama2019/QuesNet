from flask import Flask, render_template, request
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("twmkn9/bert-base-uncased-squad2")
model = torch.load(r"C:\Users\X1 Carbon\Downloads\finetunedmodel", map_location=torch.device('cpu'))
#model = AutoModelForQuestionAnswering.from_pretrained("twmkn9/bert-base-uncased-squad2")
def extract_text_from_pdf(pdf_file):
    # Ensure the uploaded file is a PDF
    if not pdf_file.filename.endswith('.pdf'):
        return "Invalid file format. Please upload a PDF file."

    # Save the uploaded file temporarily
    temp_file_path = "temp.pdf"
    pdf_file.save(temp_file_path)

    # Open the PDF file
    pdf_document = fitz.open(temp_file_path)

    # Initialize an empty string to store the extracted text
    text_content = ""

    # Iterate through pages and extract text
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        text_content += page.get_text()

    # Close the PDF file
    pdf_document.close()

    # Remove the temporary file
    # os.remove(temp_file_path)

    return text_content

def extract_text_from_excel(excel_file):
    # Ensure the uploaded file is an Excel file
    if not excel_file.filename.endswith(('.xls', '.xlsx')):
        return "Invalid file format. Please upload an Excel file."

    # Read the Excel file using pandas
    df = pd.read_excel(excel_file)

    # Combine all the text from different columns into one string
    text_content = ' '.join(str(cell) for col in df for cell in df[col])

    return text_content

def predict(context, query):
    # Print the types of query and context
    print("Query type:", type(query))
    print("Context type:", type(context))

    # Tokenize the entire context
    inputs = tokenizer.encode_plus(query, context, return_tensors='pt', max_length=512, truncation=True)

    # Call the model for prediction
    outputs = model(**inputs)

    # Retrieve start and end logits from the model outputs
    start_logits, end_logits = outputs.start_logits, outputs.end_logits

    # Find the answer span with the highest logits
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1

    # Convert the predicted span to text
    predicted_answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
    )

    return predicted_answer

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Check the file type
        if file.filename.endswith('.pdf'):
            # Extract text content from the PDF file
            file_content = extract_text_from_pdf(file)
        elif file.filename.endswith(('.txt', '.text')):
            # Extract text content from the text file
            file_content = file.read().decode('utf-8')
        elif file.filename.endswith(('.xls', '.xlsx')):
            # Extract text content from the Excel file
            file_content = extract_text_from_excel(file)
        else:
            # Handle other file types or raise an error
            return "Invalid file format. Please upload a PDF, text, or Excel file."

        # Get the question from the form
        question = request.form['question']

        # Get the predicted answer
        predicted_answer = predict(file_content, question)

        return render_template('result.html', question=question, predicted_answer=predicted_answer)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
