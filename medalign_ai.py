
# MedAlignAI - NLP-Driven Curriculum Assessment Alignment Platform (Enhanced Version)

import os
import pandas as pd
import docx2txt
import textract
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from fpdf import FPDF
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import io

class MedAlignAI:
    def __init__(self):
        self.learning_outcomes = []
        self.lecture_texts = []
        self.assessment_questions = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_text_from_uploaded_files(self, files, tag: str) -> List[str]:
        texts = []
        for uploaded_file in files:
            if tag not in uploaded_file.name.lower():
                continue
            suffix = uploaded_file.name.lower().split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                if suffix == "docx":
                    text = docx2txt.process(tmp_path)
                elif suffix in ["pdf", "txt"]:
                    text = textract.process(tmp_path).decode("utf-8")
                else:
                    continue
                texts.append(text)
            except Exception as e:
                texts.append("")
                print(f"Failed to extract text from {uploaded_file.name}: {e}")

        return texts

    def load_learning_outcomes(self, outcomes: List[str]):
        self.learning_outcomes = outcomes

    def load_tagged_files(self, uploaded_files):
        self.lecture_texts = self.extract_text_from_uploaded_files(uploaded_files, tag="lecture")
        self.assessment_questions = self.extract_text_from_uploaded_files(uploaded_files, tag="exam")

    def semantic_match(self, source_list: List[str], target_list: List[str], threshold: float = 0.7) -> Dict[str, List[str]]:
        source_emb = self.model.encode(source_list, convert_to_tensor=True)
        target_emb = self.model.encode(target_list, convert_to_tensor=True)
        matches = {}
        for i, src in enumerate(source_list):
            sim_scores = util.cos_sim(source_emb[i], target_emb)[0]
            matched = [target_list[j] for j, score in enumerate(sim_scores) if score >= threshold]
            matches[src] = matched
        return matches

    def analyze_alignment(self) -> pd.DataFrame:
        lecture_matches = self.semantic_match(self.learning_outcomes, self.lecture_texts)
        assessment_matches = self.semantic_match(self.learning_outcomes, self.assessment_questions)

        data = []
        for lo in self.learning_outcomes:
            taught = "Yes" if lecture_matches[lo] else "No"
            tested = "Yes" if assessment_matches[lo] else "No"
            data.append({
                "Learning Outcome": lo,
                "Taught": taught,
                "Tested": tested,
                "Taught and Tested": "Yes" if taught == "Yes" and tested == "Yes" else "No",
                "Tested but not Taught": "Yes" if taught == "No" and tested == "Yes" else "No",
                "Taught but not Tested": "Yes" if taught == "Yes" and tested == "No" else "No"
            })

        return pd.DataFrame(data)

    def get_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        total = len(df)
        return {
            "Total Learning Outcomes": total,
            "Taught and Tested (%)": round(100 * (df['Taught and Tested'] == "Yes").sum() / total, 2),
            "Tested but not Taught (%)": round(100 * (df['Tested but not Taught'] == "Yes").sum() / total, 2),
            "Taught but not Tested (%)": round(100 * (df['Taught but not Tested'] == "Yes").sum() / total, 2)
        }

    def generate_pdf_report(self, summary: Dict[str, float], file_path: str):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Curriculum Alignment Summary Report", ln=True, align='C')
        for k, v in summary.items():
            pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)
        pdf.output(file_path)

    def export_to_excel(self, df: pd.DataFrame) -> io.BytesIO:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Alignment')
        output.seek(0)
        return output

    def plot_summary(self, summary: Dict[str, float]):
        categories = list(summary.keys())[1:]  # skip total count
        values = [summary[k] for k in categories]

        fig, ax = plt.subplots()
        sns.barplot(x=categories, y=values, ax=ax)
        ax.set_title("Curriculum Alignment Summary")
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Streamlit UI

def main():
    st.set_page_config(page_title="MedAlignAI", layout="wide")
    st.title("🧠 MedAlignAI - Curriculum Assessment Alignment Platform")
    st.markdown("Upload tagged documents and paste your learning outcomes below.")

    lo_input = st.text_area("📋 Paste Learning Outcomes (one per line):")
    uploaded_files = st.file_uploader("📂 Upload Files (name must include 'lecture', 'exam', or 'blueprint')", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if st.button("🚀 Run Alignment"):
        los = [line.strip() for line in lo_input.strip().splitlines() if line.strip()]

        if not los:
            st.error("❗ Please enter at least one learning outcome.")
            return
        if not uploaded_files:
            st.error("❗ Please upload at least one file.")
            return

        ai = MedAlignAI()
        ai.load_learning_outcomes(los)
        ai.load_tagged_files(uploaded_files)

        result_df = ai.analyze_alignment()
        summary = ai.get_summary(result_df)

        st.subheader("📊 Alignment Results")
        st.dataframe(result_df)

        st.subheader("📈 Alignment Summary")
        for k, v in summary.items():
            st.write(f"{k}: {v}%")

        ai.plot_summary(summary)

        excel_data = ai.export_to_excel(result_df)
        st.download_button("📥 Download Excel Report", data=excel_data, file_name="alignment_report.xlsx")

        if st.button("📄 Download PDF Summary"):
            ai.generate_pdf_report(summary, "alignment_summary.pdf")
            with open("alignment_summary.pdf", "rb") as f:
                st.download_button("Download PDF Summary", f, file_name="alignment_summary.pdf")

if __name__ == "__main__":
    main()
