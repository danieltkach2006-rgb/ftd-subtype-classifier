"""
FTD Subtype Classification Tool — Clinical Decision Support (Research Prototype)
Run: python app.py
Open: http://localhost:5000
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

def generate_patient(subtype):
    if subtype == 'TDP-43':
        age_onset = np.random.normal(58, 6)
        behavior_first = np.random.choice([1, 0], p=[0.6, 0.4])
        language_first = np.random.choice([1, 0], p=[0.3, 0.7])
        motor_symptoms = np.random.choice([1, 0], p=[0.35, 0.65])
        family_history = np.random.choice([1, 0], p=[0.4, 0.6])
        mutation_TARDBP = np.random.choice([1, 0], p=[0.55, 0.45])
        mutation_MAPT = 0
        mutation_GRN = np.random.choice([1, 0], p=[0.5, 0.5])
        mutation_C9orf72 = np.random.choice([1, 0], p=[0.5, 0.5])
        mutation_FUS = 0
    elif subtype == 'Tau':
        age_onset = np.random.normal(55, 7)
        behavior_first = np.random.choice([1, 0], p=[0.7, 0.3])
        language_first = np.random.choice([1, 0], p=[0.2, 0.8])
        motor_symptoms = np.random.choice([1, 0], p=[0.1, 0.9])
        family_history = np.random.choice([1, 0], p=[0.45, 0.55])
        mutation_TARDBP = 0
        mutation_MAPT = np.random.choice([1, 0], p=[0.85, 0.15])
        mutation_GRN = 0
        mutation_C9orf72 = 0
        mutation_FUS = 0
    else:
        age_onset = np.random.normal(38, 6)
        behavior_first = np.random.choice([1, 0], p=[0.65, 0.35])
        language_first = np.random.choice([1, 0], p=[0.15, 0.85])
        motor_symptoms = np.random.choice([1, 0], p=[0.4, 0.6])
        family_history = np.random.choice([1, 0], p=[0.25, 0.75])
        mutation_TARDBP = 0
        mutation_MAPT = 0
        mutation_GRN = 0
        mutation_C9orf72 = 0
        mutation_FUS = np.random.choice([1, 0], p=[0.85, 0.15])
    return {
        'age_onset': max(20, age_onset), 'behavior_first': behavior_first,
        'language_first': language_first, 'motor_symptoms': motor_symptoms,
        'family_history': family_history, 'mutation_TARDBP': mutation_TARDBP,
        'mutation_MAPT': mutation_MAPT, 'mutation_GRN': mutation_GRN,
        'mutation_C9orf72': mutation_C9orf72, 'mutation_FUS': mutation_FUS,
        'subtype': subtype
    }

N_PER_CLASS = 200
rows = []
for subtype in ['TDP-43', 'Tau', 'FUS']:
    for _ in range(N_PER_CLASS):
        rows.append(generate_patient(subtype))

df = pd.DataFrame(rows)
features = ['age_onset', 'behavior_first', 'language_first', 'motor_symptoms',
            'family_history', 'mutation_TARDBP', 'mutation_MAPT', 'mutation_GRN',
            'mutation_C9orf72', 'mutation_FUS']

X = df[features]
y = df['subtype']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, pred)
print(f"Model trained. Test accuracy: {test_accuracy:.1%}")
print(classification_report(y_test, pred))

# ── Flask App ──

from flask import Flask, request, render_template_string
app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>FTD Subtype Classification Tool</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #eef1f5; margin: 0; padding: 40px 20px; color: #1a1a1a; }
  .container { max-width: 620px; margin: 0 auto; }
  .card { background: #fff; padding: 30px; border-radius: 10px; border: 1px solid #d0d5dd; margin-bottom: 24px; }
  h1 { font-size: 21px; color: #1a1a1a; margin-bottom: 2px; }
  .subtitle { color: #555; font-size: 13px; margin-bottom: 18px; border-bottom: 2px solid #0057a8; padding-bottom: 12px; }
  .section-title { font-size: 15px; font-weight: bold; color: #0057a8; margin-top: 22px; margin-bottom: 6px; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; }
  label { display: block; margin-top: 16px; font-weight: 600; font-size: 13px; }
  .desc { font-size: 11px; color: #555; margin-top: 2px; font-style: italic; }
  select, input[type=number] { width: 100%; padding: 9px; margin-top: 5px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; background: #fafafa; }
  button { margin-top: 24px; width: 100%; padding: 13px; background: #0057a8; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 15px; font-weight: bold; letter-spacing: 0.3px; }
  button:hover { background: #003d75; }
  .result-card { background: #f0f6ff; border: 2px solid #0057a8; border-radius: 10px; padding: 24px; }
  .result-card h2 { color: #0057a8; margin-top: 0; font-size: 19px; }
  .result-summary { font-size: 13px; color: #333; margin-bottom: 16px; }
  .prob-label { font-size: 13px; font-weight: 600; margin-top: 4px; }
  .bar-bg { background: #e0e0e0; border-radius: 4px; height: 26px; margin: 4px 0 14px 0; overflow: hidden; }
  .bar-fill { height: 100%; color: white; font-size: 12px; text-align: right; padding-right: 8px; line-height: 26px; font-weight: bold; border-radius: 4px; min-width: 2%; }
  .bg-tdp43 { background: #0057a8; }
  .bg-tau { background: #2e8b57; }
  .bg-fus { background: #c0392b; }
  .disclaimer { font-size: 11px; color: #888; margin-top: 18px; border-top: 1px solid #e0e0e0; padding-top: 10px; }
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>Frontotemporal Dementia &mdash; Subtype Classification Tool</h1>
    <div class="subtitle">Clinical Decision Support System (Research Prototype) &nbsp;|&nbsp; Model Accuracy: {{ accuracy }}</div>
    <form method="POST">

      <div class="section-title">Clinical Presentation</div>

      <label>Patient Age at Symptom Onset</label>
      <div class="desc">Age when the patient or family first noticed cognitive, behavioral, or language changes</div>
      <input type="number" name="age_onset" placeholder="Enter age (e.g. 55)" required>

      <label>Primary Presenting Symptom: Behavioral or Personality Changes</label>
      <div class="desc">Examples: social disinhibition, apathy, loss of empathy, compulsive behaviors, poor judgment</div>
      <select name="behavior_first" required>
        <option value="" selected disabled>&mdash; Select &mdash;</option>
        <option value="1">Yes &mdash; behavioral/personality changes were the primary presenting symptom</option>
        <option value="0">No &mdash; another symptom type was primary</option>
      </select>

      <label>Primary Presenting Symptom: Language Difficulty</label>
      <div class="desc">Examples: progressive difficulty with word-finding, speech production, sentence comprehension, or naming</div>
      <select name="language_first" required>
        <option value="" selected disabled>&mdash; Select &mdash;</option>
        <option value="1">Yes &mdash; language impairment was the primary presenting symptom</option>
        <option value="0">No &mdash; language was not the primary symptom</option>
      </select>

      <label>Motor Neuron Symptoms Present</label>
      <div class="desc">Examples: progressive muscle weakness, fasciculations, difficulty swallowing, features consistent with ALS (amyotrophic lateral sclerosis)</div>
      <select name="motor_symptoms" required>
        <option value="" selected disabled>&mdash; Select &mdash;</option>
        <option value="1">Yes &mdash; motor neuron involvement observed</option>
        <option value="0">No &mdash; no motor neuron symptoms</option>
      </select>

      <label>Family History of Neurodegenerative Disease</label>
      <div class="desc">First- or second-degree relative diagnosed with FTD, ALS, Alzheimer's, or other dementia</div>
      <select name="family_history" required>
        <option value="" selected disabled>&mdash; Select &mdash;</option>
        <option value="1">Yes &mdash; positive family history</option>
        <option value="0">No &mdash; no known family history</option>
      </select>

      <div class="section-title">Genetic Testing Results</div>

      <label>TARDBP Gene Mutation</label>
      <div class="desc">TARDBP encodes the TDP-43 protein. Mutations are associated with TDP-43 proteinopathy and may co-occur with ALS.</div>
      <select name="mutation_TARDBP" required>
        <option value="" selected disabled>&mdash; Select &mdash;</option>
        <option value="1">Mutation detected</option>
        <option value="0">No mutation / Not tested</option>
      </select>

      <label>MAPT Gene Mutation</label>
      <div class="desc">MAPT encodes the Tau protein. Mutations are strongly associated with Tau-type FTD (Pick's disease) and behavioral-variant presentations.</div>
      <select name="mutation_MAPT" required>
        <option value="" selected disabled>&mdash; Select &mdash;</option>
        <option value="1">Mutation detected</option>
        <option value="0">No mutation / Not tested</option>
      </select>

      <label>GRN Gene Mutation (Progranulin)</label>
      <div class="desc">GRN mutations cause progranulin deficiency and are associated with TDP-43 proteinopathy, often presenting with language-variant FTD.</div>
      <select name="mutation_GRN" required>
        <option value="" selected disabled>&mdash; Select &mdash;</option>
        <option value="1">Mutation detected</option>
        <option value="0">No mutation / Not tested</option>
      </select>

      <label>C9orf72 Gene Repeat Expansion</label>
      <div class="desc">C9orf72 hexanucleotide repeat expansion is the most common genetic cause of both FTD and ALS. Associated with TDP-43 pathology.</div>
      <select name="mutation_C9orf72" required>
        <option value="" selected disabled>&mdash; Select &mdash;</option>
        <option value="1">Expansion detected</option>
        <option value="0">No expansion / Not tested</option>
      </select>

      <label>FUS Gene Mutation</label>
      <div class="desc">FUS mutations are rare and associated with FUS-type FTD, typically presenting at a younger age (often under 45).</div>
      <select name="mutation_FUS" required>
        <option value="" selected disabled>&mdash; Select &mdash;</option>
        <option value="1">Mutation detected</option>
        <option value="0">No mutation / Not tested</option>
      </select>

      <button type="submit">Run Subtype Classification</button>
    </form>
  </div>

  {% if result %}
  <div class="result-card">
    <h2>Classification Result: {{ result.predicted }} Proteinopathy</h2>
    <div class="result-summary">Based on the provided clinical and genetic profile, the model predicts <strong>{{ result.predicted }}</strong> as the most likely underlying protein pathology.</div>
    {% for k, v in result.sorted_probs %}
    <div class="prob-label">{{ k }} Proteinopathy: {{ "%.1f"|format(v*100) }}%</div>
    <div class="bar-bg"><div class="bar-fill bg-{{ k|lower|replace('-','') }}" style="width:{{ [v*100, 2]|max }}%">{{ "%.1f"|format(v*100) }}%</div></div>
    {% endfor %}
  </div>
  {% endif %}

  <p class="disclaimer">This is a research prototype trained on synthetic data calibrated to published genotype-phenotype associations. It is not a validated clinical diagnostic tool and should not be used for medical decision-making without physician oversight and confirmatory testing.</p>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        patient = {
            'age_onset': float(request.form['age_onset']),
            'behavior_first': int(request.form['behavior_first']),
            'language_first': int(request.form['language_first']),
            'motor_symptoms': int(request.form['motor_symptoms']),
            'family_history': int(request.form['family_history']),
            'mutation_TARDBP': int(request.form['mutation_TARDBP']),
            'mutation_MAPT': int(request.form['mutation_MAPT']),
            'mutation_GRN': int(request.form['mutation_GRN']),
            'mutation_C9orf72': int(request.form['mutation_C9orf72']),
            'mutation_FUS': int(request.form['mutation_FUS']),
        }
        X_input = pd.DataFrame([patient])[features]
        probs = model.predict_proba(X_input)[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, probs))
        predicted = max(prob_dict, key=prob_dict.get)
        sorted_probs = sorted(prob_dict.items(), key=lambda x: -x[1])
        result = {'predicted': predicted, 'sorted_probs': sorted_probs}

    return render_template_string(PAGE, result=result, accuracy=f"{test_accuracy:.1%}")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("FTD Subtype Classification Tool")
    print(f"Model accuracy: {test_accuracy:.1%}")
    print("Open your browser to: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000)
