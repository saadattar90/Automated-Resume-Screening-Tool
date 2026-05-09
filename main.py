import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Job Description
with open("data/job_description.txt", "r", encoding="utf-8") as f:
    job_description = f.read()

resume_folder = "resumes"

resume_data = []
resume_names = []

# Read resumes
for file in os.listdir(resume_folder):
    if file.endswith(".txt"):
        path = os.path.join(resume_folder, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

            resume_names.append(file)
            resume_data.append(text)

# TF-IDF Vectorization
documents = [job_description] + resume_data

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(documents)

# Cosine Similarity
similarity_scores = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)

scores = similarity_scores[0]

# Create DataFrame
df = pd.DataFrame({
    "Resume": resume_names,
    "Score": scores
})

# Ranking
df = df.sort_values(by="Score", ascending=False)

# Shortlist
df["Status"] = df["Score"].apply(
    lambda x: "Shortlisted" if x > 0.2 else "Rejected"
)

print("\n===== RESUME SCREENING RESULTS =====\n")

print(df)

# Save CSV
os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/resume_ranking.csv", index=False)

# Graph
plt.figure(figsize=(8,5))

plt.bar(df["Resume"], df["Score"])

plt.title("Resume Matching Scores")

plt.xlabel("Resumes")

plt.ylabel("Score")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("outputs/resume_scores.png")

print("\nCSV Report Saved Successfully!")

print("Graph Saved Successfully!")