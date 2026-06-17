import os
import json
import subprocess
from openai import OpenAI
import getpass
import time

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = getpass.getpass("Please enter your OpenAI API key: ")

client = OpenAI(api_key=api_key)

JD_DIR = "JDs_Detailed"
OUTPUT_DIR = "Generated_CVs"

TEMPLATE = r"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{geometry}}
\geometry{{a4paper, margin=1in}}
\usepackage{{enumitem}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}

\pagestyle{{empty}}

\begin{{document}}

\begin{{center}}
    \textbf{{\Large {name}}}\\[0.5em]
    | {phone} | {address} | {email} | \href{{{github}}}{{Github}}
\end{{center}}

\vspace{{1em}}
\noindent \textbf{{\large Summary}}\\
\rule{{\textwidth}}{{0.4pt}}
{summary}

\vspace{{1em}}
\noindent \textbf{{\large Education}}\\
\rule{{\textwidth}}{{0.4pt}}
\textbf{{{university}}} \hfill {uni_location}\\
{degree}. GPA: \fbox{{{gpa}}} \hfill {uni_years}\\
\textbf{{IELTS Certificate}}\\
Score \fbox{{{ielts}}}

\vspace{{1em}}
\noindent \textbf{{\large Experience}}\\
\rule{{\textwidth}}{{0.4pt}}
{experience}

\vspace{{1em}}
\noindent \textbf{{\large Project}}\\
\rule{{\textwidth}}{{0.4pt}}
{projects}

\vspace{{1em}}
\noindent \textbf{{\large Skills}}\\
\rule{{\textwidth}}{{0.4pt}}
{skills}

\end{{document}}
"""

def generate_content(jd_text, job_title, category):
    prompt = f"""
    You are an expert CV writer.
    Based on the following Job Description for '{job_title}' in the category '{category}',
    generate a fake junior/intern candidate's CV data that perfectly matches the JD requirements.
    
    IMPORTANT: Do not use special LaTeX characters like '&', '%', '$', '#', '_', '{{', '}}', '~', '^', '\\' without proper escaping in text fields (e.g., escape '&' as '\\&').

    The format must be JSON with the following keys:
    - name: A fake Vietnamese name (e.g., "NGUYEN VAN A").
    - phone: A fake phone number (e.g., "84 912345678").
    - address: A fake address in Vietnam (e.g., "District 1, Ho Chi Minh City").
    - email: A fake email (e.g., "nguyenvana@gmail.com").
    - github: A fake github link (e.g., "https://github.com/nguyenvana").
    - summary: A short 3-4 sentence paragraph summarizing their background (student/recent grad) and projects/internships matching the JD.
    - university: Fake university name in English (e.g., "University of Economics").
    - uni_location: Location (e.g., "Ho Chi Minh, VNU").
    - degree: Degree (e.g., "Bachelor in Marketing").
    - gpa: "3.5/4.0" or similar.
    - uni_years: "2021 - Now".
    - ielts: "6.5" or similar.
    - experience: A string in LaTeX format for their experience. Escape any special characters. Use this exact structure:
      \\textbf{{Company Name}} \\hfill Month Year - Month Year\\\\
      \\textbf{{Tiltle: Job Title Intern/Junior}}\\\\
      \\textbf{{Project: Project Name}} - Role: Role Name \\hfill Year
      \\begin{{itemize}}[leftmargin=*]
          \\item Description 1
          \\item Description 2
      \\end{{itemize}}
    - projects: A string in LaTeX format for 2 personal projects. Escape special characters. Use this exact structure:
      \\textbf{{Project Name}} \\hfill Year
      \\begin{{itemize}}[leftmargin=*]
          \\item Description 1
          \\item Description 2
      \\end{{itemize}}
      \\vspace{{0.5em}}
      \\textbf{{Project 2 Name}} \\hfill Year
      \\begin{{itemize}}[leftmargin=*]
          \\item Description 1
      \\end{{itemize}}
    - skills: A string of skills separated by | (e.g., "Skill 1 | Skill 2 | Skill 3").

    JD Text:
    {jd_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output pure JSON. Do not include markdown codeblocks like ```json...``` in the output."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        response_format={ "type": "json_object" }
    )
    
    return json.loads(response.choices[0].message.content)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for root, dirs, files in os.walk(JD_DIR):
        for file in files:
            if file.endswith(".md"):
                category = os.path.basename(root)
                jd_path = os.path.join(root, file)
                job_title = file.replace(".md", "")
                
                print(f"Processing: {job_title} in {category}")
                
                with open(jd_path, 'r', encoding='utf-8') as f:
                    jd_text = f.read()

                # Generate content
                try:
                    data = generate_content(jd_text, job_title, category)
                except Exception as e:
                    print(f"Failed to generate for {job_title}: {e}")
                    continue
                
                # Fill template
                try:
                    tex_content = TEMPLATE.format(
                        name=data.get('name', 'N/A'),
                        phone=data.get('phone', 'N/A'),
                        address=data.get('address', 'N/A'),
                        email=data.get('email', 'N/A'),
                        github=data.get('github', 'N/A'),
                        summary=data.get('summary', 'N/A'),
                        university=data.get('university', 'N/A'),
                        uni_location=data.get('uni_location', 'N/A'),
                        degree=data.get('degree', 'N/A'),
                        gpa=data.get('gpa', 'N/A'),
                        uni_years=data.get('uni_years', 'N/A'),
                        ielts=data.get('ielts', 'N/A'),
                        experience=data.get('experience', ''),
                        projects=data.get('projects', ''),
                        skills=data.get('skills', '')
                    )
                except Exception as e:
                    print(f"Template formatting failed for {job_title}: {e}")
                    continue
                
                # Save tex
                out_cat_dir = os.path.join(OUTPUT_DIR, category)
                os.makedirs(out_cat_dir, exist_ok=True)
                
                tex_path = os.path.join(out_cat_dir, f"{job_title}.tex")
                with open(tex_path, 'w', encoding='utf-8') as f:
                    f.write(tex_content)
                
                # Compile PDF
                print(f"Compiling PDF for {job_title}...")
                # Run twice to ensure references/formatting
                subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', out_cat_dir, tex_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', out_cat_dir, tex_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Clean up auxiliary files
                for ext in ['.aux', '.log', '.out']:
                    aux_file = os.path.join(out_cat_dir, f"{job_title}{ext}")
                    if os.path.exists(aux_file):
                        os.remove(aux_file)
                
                time.sleep(0.5)

    print("All done! Check Generated_CVs directory.")

if __name__ == "__main__":
    main()
