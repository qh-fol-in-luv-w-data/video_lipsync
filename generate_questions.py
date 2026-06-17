import os
import subprocess
from openai import OpenAI
import time

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

JD_DIR = "JDs_Detailed"
OUTPUT_DIR = "Generated_Questions"

def generate_questions(jd_text, job_title):
    prompt = f"""
    You are an expert technical interviewer/HR recruiter.
    Based on the following Job Description (JD) for '{job_title}', generate exactly 6 highly relevant interview questions (in Vietnamese) that assess the candidate's fit for this specific role. The questions should cover:
    - 2 questions on Hard Skills/Technical knowledge mentioned in the JD.
    - 2 questions on Soft Skills/Scenario based situations.
    - 2 questions on their past experience/projects.

    JD Details:
    {jd_text}
    
    Output strictly as a Markdown list:
    1. **[Technical]** ...
    2. **[Technical]** ...
    3. **[Soft Skill]** ...
    ...
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    
    return response.choices[0].message.content

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for root, dirs, files in os.walk(JD_DIR):
        for file in files:
            if file.endswith(".md"):
                category = os.path.basename(root)
                jd_path = os.path.join(root, file)
                job_title = file.replace(".md", "")
                
                print(f"Processing Questions: {job_title} in {category}")
                
                with open(jd_path, 'r', encoding='utf-8') as f:
                    jd_text = f.read()

                try:
                    questions_text = generate_questions(jd_text, job_title)
                except Exception as e:
                    print(f"Failed to generate for {job_title}: {e}")
                    continue
                
                out_cat_dir = os.path.join(OUTPUT_DIR, category)
                os.makedirs(out_cat_dir, exist_ok=True)
                
                out_path = os.path.join(out_cat_dir, f"{job_title}_Questions.md")
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Câu hỏi phỏng vấn cho vị trí: {job_title.replace('_', ' ')}\n\n")
                    f.write(questions_text)
                
                time.sleep(0.5)

    print("All questions generated successfully in Generated_Questions directory.")

if __name__ == "__main__":
    main()
