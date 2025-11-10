import streamlit as st
import PyPDF2
import openai
import os
import io
from dotenv import load_dotenv

load_dotenv()

def main():
    st.set_page_config(page_title="AI Resume Critiquer", page_icon=":memo:")
    st.title("AI Resume Critiquer")
    st.markdown("Upload your resume in PDF format and get instant feedback to improve it!")

    OPEN_API_KEY  = os.getenv("OPEN_API_KEY")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf","txt"])
    job_role      = st.text_input("Enter the job role you are applying for (optional):")
    analyze_button = st.button("Analyze Resume")


    def extract_text_from_pdf(pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text


    def extract_text_from_file(uploaded_file):
        if uploaded_file.type == "application/pdf":
            return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
        return uploaded_file.read().decode("utf-8")


    if analyze_button and uploaded_file is not None:

        try:
            file_content = extract_text_from_file(uploaded_file)

            if file_content.strip() == "":
                st.error("The uploaded file is empty or could not be read. Please upload a valid PDF or text file.")
                st.stop()

            prompt = f"""Please analyze this resume and provide constructive feedback. 
            Focus on the following aspects:
            1. Content clarity and impact
            2. Skills presentation
            3. Experience descriptions
            4. Specific improvements for {job_role if job_role else 'general job applications'}
            
            Resume content:
            {file_content}
            
            Please provide your analysis in a clear, structured format with specific recommendations."""
            
            client = openai.OpenAI(api_key=OPEN_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides resume critiques."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )


            st.markdown("### Resume Analysis")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()  
