import tkinter as tk
import csv
from langdetect import detect
from nltk.corpus import stopwords
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tkinter import filedialog
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PlagiarismDetector:
    def __init__(self):
        self.documents = {}
        self.vectorizer = TfidfVectorizer()

    @staticmethod
    def preprocess_text(text, language):
        if language == 'english':
            stop_words = set(stopwords.words('english'))
            words = text.split()
            words = [word for word in words if word.lower() not in stop_words]
            return ' '.join(words)
        return text

    def add_document(self, text, language):
        self.documents[language] = text

    def build_tfidf_matrix(self):
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents.values())

    def calculate_similarity(self, text, language):
        if 'tfidf_matrix' not in self.__dict__:
            self.build_tfidf_matrix()

        preprocessed_text = self.preprocess_text(text, language)
        text_vector = self.vectorizer.transform([preprocessed_text])
        similarities = cosine_similarity(text_vector, self.tfidf_matrix)
        return similarities

    def check_plagiarism(self, text, language, threshold=0.7):
        similarities = self.calculate_similarity(text, language)
        similar_docs = []

        for lang, sim_scores in zip(self.documents.keys(), similarities):
            for i, sim_score in enumerate(sim_scores):
                if sim_score > threshold:
                    similar_docs.append((self.documents[lang], sim_score))

        return similar_docs


class PlagiarismApp:
    def generate_pdf_report(self):
        result_text = self.result_text.get(1.0, tk.END)
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            c = canvas.Canvas(file_path, pagesize=letter)
            c.drawString(100, 750, "Plagiarism Detection Report")
            c.drawString(100, 730, "-" * 30)
            c.drawString(100, 700, result_text)
            c.save()

    def generate_csv_report(self):
        result_text = self.result_text.get(1.0, tk.END)
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            with open(file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Plagiarism Detection Report"])
                writer.writerow(["Result"])
                writer.writerow([""])
                writer.writerow([result_text])

    def __init__(self, root):
        self.root = root
        self.root.title("Plagiarism Detector")

        self.detector = PlagiarismDetector()

        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=100, height=20)
        self.text_area.pack(padx=10, pady=10)

        self.add_button = tk.Button(self.root, text="Add File", command=self.add_file)
        self.add_button.pack()

        self.check_button = tk.Button(self.root, text="Check Plagiarism", command=self.check_plagiarism)
        self.check_button.pack()

        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset)
        self.reset_button.pack()

        self.result_text = tk.Text(self.root, wrap=tk.WORD, width=100, height=20)
        self.result_text.pack(padx=10, pady=10)

        self.pdf_button = tk.Button(self.root, text="PDF", command=self.generate_pdf_report)

        self.csv_button = tk.Button(self.root, text="CSV", command=self.generate_csv_report)

        self.button_frame = tk.Frame(self.root)
        self.pdf_button.pack(side=tk.LEFT, padx=200, pady=10)
        self.csv_button.pack(side=tk.LEFT, padx=255, pady=10)
        self.button_frame.pack()

        self.button_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def add_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                language = detect(content)
                self.text_area.insert(tk.END, content + '\n')
                self.detector.add_document(content, language)
                self.result_text.delete(1.0, tk.END)

    def check_plagiarism(self):
        query_text = self.text_area.get(1.0, tk.END).strip()
        if query_text:
            language = detect(query_text)
            similar_documents = self.detector.check_plagiarism(query_text, language)

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Similar Documents:\n")
            for doc, sim_score in similar_documents:
                dissimilarity_percentage = (1 - sim_score) * 100

                if 0 <= dissimilarity_percentage <= 25:
                    result = "PLAGIARISM!"
                elif 25 < dissimilarity_percentage <= 50:
                    result = "SUSPECTED!"
                elif 50 < dissimilarity_percentage <= 75:
                    result = "MAYBE!"
                else:
                    result = "NO PROBLEM!"

                self.result_text.insert(tk.END, f"Dissimilarity Percentage: {dissimilarity_percentage:.2f}%\n\n")
                self.result_text.insert(tk.END, f"Result: {result}\n")

    def reset(self):
        self.text_area.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.detector = PlagiarismDetector()


if __name__ == "__main__":
    PlagiarismApp(tk.Tk()).root.mainloop()
