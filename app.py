import streamlit as st
import base64
from jinja2 import Environment, FileSystemLoader
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import pdfkit
import os
import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import pyrebase

processor = AutoProcessor.from_pretrained("saved_model")
model = AutoModelForCausalLM.from_pretrained("saved_model")

# Firebase initialization
def initialize_firebase():
    config = {
        # Paste firebase config here..
    }

    firebase = pyrebase.initialize_app(config)
    auth = firebase.auth()
    db = firebase.database()
    storage = firebase.storage()

    return auth, db, storage

# Firebase login function
def login(auth, email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        user_info = auth.get_account_info(user['idToken'])
        user_id = user_info['users'][0]['localId']
        st.session_state.logged_in = True
        st.session_state.user_email = email
        st.session_state.user_id = user_id  # Set user_id in session state
        return user
    except pyrebase.pyrebase.HTTPError as e:
        st.error("Login failed. Invalid email or password.")



# Firebase signup function
def signup(auth, email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        return user
    except pyrebase.pyrebase.HTTPError as e:
        st.error("Signup failed. Email already exists.")

# Function to generate report
# Function to generate report
def generate_report(db, storage):
    template_text = "report_template.html"
    env = Environment(loader=FileSystemLoader(searchpath="./"))
    
    st.header("ActivMedica")

    with st.form("my_form"):
        st.subheader("Details")
        
        form_details = st.session_state.get("form_details", {})
        name = st.text_input("Name", value=form_details.get("name", ""))
        
        # Ensure gender default value is one of the provided options
        gender_options = ["Male", "Female", "Others"]
        gender_default = form_details.get("gender", "")
        if gender_default not in gender_options:
            gender_default = gender_options[0]  # Set default to the first option if not found
        gender = st.radio("Gender", gender_options, index=gender_options.index(gender_default))
        
        age = st.text_input("Age", value=form_details.get("age", ""))
        blood_group = st.text_input("Blood Group", value=form_details.get("blood_group", ""))
        height = st.text_input("Height", value=form_details.get("height", ""))
        weight = st.text_input("Weight", value=form_details.get("weight", ""))
        phone = st.text_input("Phone no", value=form_details.get("phone", ""))
        doctor = st.text_input("Doctor's Name", value=form_details.get("doctor", ""))
        radiologist = st.text_input("Radiologist's Name", value=form_details.get("radiologist", ""))
        uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

        # Input field for custom file name
        custom_filename = st.text_input("Enter custom file name (without extension)", "report")

        # Persist form details in session state
        st.session_state.form_details = {
            "name": name,
            "gender": gender,
            "age": age,
            "blood_group": blood_group,
            "height": height,
            "weight": weight,
            "phone": phone,
            "doctor": doctor,
            "radiologist": radiologist,
            "uploaded_file": uploaded_file
        }

        if st.form_submit_button("Generate Report"):
            pdf_text = ""

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image.save("input.jpg")
                with st.spinner("Generating caption.."):
                    caption = generate_caption(image)

            # Generate PDF file name with custom name and timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{custom_filename}_{timestamp}.pdf"

            # Load template
            template = env.get_template(template_text)

            # Render template with user data
            data = {
                "patient_name": name,
                "age": age,
                "blood_group": blood_group,
                "patient_phone": phone,
                "patient_height": height,
                "patient_weight": weight,
                "radio_name": radiologist,
                "doc_name": doctor,
                "patient_gender": gender,
                "diagnosis": caption if caption else "No diagnosis available"
            }
            rendered_html = template.render(data)
            with open("output.html", "w", encoding="utf-8") as f:
                f.write(rendered_html)
            pdfkit.from_file("output.html", filename, options={"enable-local-file-access": "", "enable-javascript": ""})

            st.session_state.filename = filename

            # Upload PDF report to Firebase Storage
            storage.child(filename).put(filename)

            # Get download URL of the uploaded PDF
            pdf_url = storage.child(filename).get_url(None)

            # Store PDF URL in database along with user ID
            user_id = st.session_state.user_id
            db.child("reports").child(user_id).push({
                "name": name,
                "pdf_url": pdf_url
            })

            # Create download link
            with open(filename, "rb") as f:
                pdf_data = f.read()
                base64_data = base64.b64encode(pdf_data).decode("utf-8")
                st.session_state.pdf_data = base64_data
                download_link = f"""
                <a href="data:application/octet-stream;base64,{st.session_state.pdf_data}" download="{filename}" style= "color: white;" class="button">
                Download PDF</a>

                <style>
                .button {{
                background-color: green;
                border: none; 
                color: white; 
                padding: 6px 18px; 
                text-align: center; 
                text-decoration: none; 
                display: inline-block;
                font-size: 16px; 
                margin: 4px 2px; 
                cursor: pointer; 
                border-radius: 6px;
                }}

                .button:hover {{
                background-color: grey;
                text-decoration: none;
                color: white;
                }}
                </style>
                """
                st.markdown(download_link, unsafe_allow_html=True)

            with open(filename, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()

            # Store pdf_text in session state
            st.session_state.pdf_text = pdf_text
            if "analyzed" in st.session_state:
                st.session_state.pop("analyzed")



# Function to generate caption for MRI image
def generate_caption(image):
    image = image.convert("RGB")  # Ensure RGB format for consistency
    image_array = np.array(image)  # Convert to a NumPy array
    inputs = processor(images=Image.fromarray(image_array), return_tensors="pt")
    image_data = inputs["pixel_values"]

    with torch.no_grad():  # Disable gradient calculation for efficiency
        generated_ids = model.generate(pixel_values=image_data, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# Function to initialize and interact with the chatbot
def chatbot():
    # Initialize Gemini-Pro    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')

    # Gemini uses 'model' for assistant; Streamlit uses 'assistant'
    def role_to_streamlit(role):
        if role == "model":
            return "assistant"
        else:
            return role

    # Add a Gemini Chat history object to Streamlit session state
    if "chat" not in st.session_state:
        st.session_state.chat = model.start_chat(history=[])

    # Display Form Title
    st.title("Chat Evolved: Introducing Gemini, Our Supercharged Chatmodel")


    if st.session_state.chat.history:
        messages_to_display = st.session_state.chat.history[1:]
        for message in messages_to_display:
            with st.chat_message(role_to_streamlit(message.role)):
                st.markdown(message.parts[0].text)

    if "pdf_text" in st.session_state:
        pdf_text = st.session_state.pdf_text
        # Analyze report only if it hasn't been analyzed before
        if "analyzed" not in st.session_state:
            new_prompt = pdf_text + "\nYou should act as doctor and give full medical report on the findings with full details"
            with st.spinner("Analyzing your report.."):
                response = st.session_state.chat.send_message(new_prompt)
            # Display last 
            with st.chat_message("assistant"):
                st.markdown(response.text)

            # Set flag to indicate report has been analyzed
            st.session_state.analyzed = True
            

        # Accept user's next message, add to context, resubmit context to Gemini
        if prompt := st.chat_input("Enter your query.."):

            # Display user's last message
            st.chat_message("user").markdown(prompt)

            with st.spinner("Thinking.."):
                response = st.session_state.chat.send_message(prompt)
                
            with st.chat_message("assistant"):
                st.markdown(response.text)


    if "pdf_data" in st.session_state:
        download_link = f"""
            <a href="data:application/octet-stream;base64,{st.session_state.pdf_data}" download="{st.session_state.filename}" style= "color: white;" class="button">
            Download PDF</a>

            <style>
            .button {{
            background-color: green;
            border: none; 
            color: white; 
            padding: 6px 18px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block;
            font-size: 16px; 
            margin: 4px 2px; 
            cursor: pointer; 
            border-radius: 6px;
            }}

            .button:hover {{
            background-color: grey;
            text-decoration: none;
            color: white;
            }}
            </style>
            """
        st.markdown(download_link, unsafe_allow_html=True)


# Main function
def main():
    auth, db, storage = initialize_firebase()

    st.set_page_config(page_title="ActivMedica(alpha)")

    st.title("")

    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")
        signup_button = st.button("Signup")

        if login_button:
            user = login(auth, email, password)
            if user:
                st.session_state.logged_in = True
                st.success("Login successful. Click on the login button again to continue.")
                st.session_state.user_email = email  # Store user's email in session state
                user_info = auth.get_account_info(user['idToken'])
                st.session_state.user_id = user_info['users'][0]['localId']  # Store user's ID in session state

        if signup_button:
            user = signup(auth, email, password)
            if user:
                st.success("Signup successful. Please login.")

    else:
        # Logout button in the sidebar
        with st.sidebar:
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.success("Logout successful. Click on the logout button again to exit.")
                return
        
            # Display current user's email and ID
            st.subheader("Account")
            st.write(f"User ID: {st.session_state.user_id}")
            st.write(f"Email: {st.session_state.user_email}")

        # Main functionality
        with st.sidebar:
            st.subheader("Navigation")
            app_mode = st.radio("Go to", ["Generate Report", "Chatbot"])

        if app_mode == "Generate Report":
            generate_report(db, storage)
        elif app_mode == "Chatbot":
            chatbot()



if __name__ == '__main__':
    main()
