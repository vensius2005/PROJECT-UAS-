import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template_string, request
import numpy as np

# ================== 1. Load Dataset ==================
df = pd.read_csv("ss.csv")  # Pastikan target sudah berupa label kelas 0 dan 1

target = "Quality of Sleep"  # Pastikan ini kelas 0 dan 1
features = [col for col in df.columns if col != target]

num_cols = df[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df[features].select_dtypes(include=["object"]).columns.tolist()

nama_kolom_id = {
    "Person ID": "ID Orang",
    "Age": "Umur",
    "Gender": "Jenis Kelamin",
    "Height": "Tinggi Badan",
    "Weight": "Berat Badan",
    "Occupation": "Pekerjaan",
    "Education": "Pendidikan",
    "Marital Status": "Status Pernikahan",
    "Sleep Duration": "Durasi Tidur",
    "Exercise Frequency": "Frekuensi Olahraga",
    "Smoking": "Merokok",
    "Alcohol Consumption": "Konsumsi Alkohol",
    "Stress Level": "Tingkat Stres",
    "Quality of Sleep": "Kualitas Tidur",
}

for col in features:
    if col not in nama_kolom_id:
        nama_kolom_id[col] = " ".join(word.capitalize() for word in col.split())

# ================== 2. Buat Pipeline ==================
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ================== 3. Split Data & Train ==================
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluasi Model (klasifikasi)
y_pred = model.predict(X_test)
print("=== Evaluasi Model ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ================== 4. Flask App ==================
app = Flask(__name__)

form_html = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Prediksi Kualitas Tidur Menggunakan Logistic Linier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body class="bg-light">
    <div class="container mt-5">
        <div class="card shadow p-4">
            <h2 class="text-center mb-4">Prediksi Kualitas Tidur Menggunakan Logistic Linier</h2>
            <form action="/predict" method="post">
                {% for col in features %}
                    <div class="mb-3">
                        <label class="form-label">{{ nama_kolom_id.get(col, col) }}</label>
                        {% if col in cat_cols %}
                            <input type="text" class="form-control" name="{{ col }}" placeholder="Masukkan kategori" required />
                        {% else %}
                            <input type="number" step="any" class="form-control" name="{{ col }}" placeholder="Masukkan angka" required />
                        {% endif %}
                    </div>
                {% endfor %}
                <button type="submit" class="btn btn-primary w-100">Prediksi</button>
            </form>
            {% if prediction is not none %}
                <div class="alert alert-info mt-4 text-center">
                    <h5>Hasil Prediksi Kualitas Tidur: {{ prediction }}</h5>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        form_html, features=features, cat_cols=cat_cols, prediction=None, nama_kolom_id=nama_kolom_id
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {}
        for col in features:
            input_data[col] = request.form[col]
        input_df = pd.DataFrame([input_data])
        for col in num_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
        pred = model.predict(input_df)[0]
        hasil = "Tidur Baik" if pred == 1 else "Tidur Buruk"
        return render_template_string(
            form_html, features=features, cat_cols=cat_cols, prediction=hasil, nama_kolom_id=nama_kolom_id
        )
    except Exception as e:
        return f"Terjadi kesalahan: {e}"

if __name__ == "__main__":
    app.run(debug=True)
