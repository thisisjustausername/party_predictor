# Party Predictor (Partei Klassifikator)
*Eine englische Version kann unter [English README](README.md) gefunden werden.*<br/><br/>
Der europaweite Trend zu rechter Politik hat auch Deutschland nicht verfehlt. Doch inwiefern kann dies an im Bundestag gehaltenen Reden nachvollzogen werden?<br/>
Wir haben tausende Reden der Parteien AfD, CDU/CSU, Bündnis_90/Die Grünen, SPD und der LINKEn analysiert. Als Baseline wurde der CHES score für die politische Richtung einer Partei verwendet. Dennoch vollzogen wir unsere Klassifikation und Analyse der sprachlichen Besonderheiten der verschiedenen Parteien unabhängig von diesem Score.<br/>
Hierzu finetuneten wir ein deutsches BERT-Modell auf Klassifikation von Reden in Parteigruppen und analysierten die Ursachen für die gegebenen Klassifikationen. Diese verglichen wir anschließend mit verschiedenen historischen Reden unterschiedlicher gemäßigter und extremen politischen Richtungen, um Gemeinsamkeiten und Unterschiede feststellen zu können. 

# Getting started
Du willst unser wunderschönes Tool verwenden ohne auch nur eine Sekunde darauf zu verschwenden, unseren supersauberen Code zu genießen. Auch wenn uns dieses Verhalten äußerst suspekt erscheint, haben wir für Dich eine kleine Übersicht an Befehlen zusammengestellt, durch die Du Dich durchhangeln kannst. So kannst Du unseren Code auch verwenden, wenn Du keine Tuten und Blasen von Programmieren hast (vorausgesetzt Du verwendest Linux und hast Python bereits installiert).<br/>

## Setup
Egal was komme, führe diesen Code aus (Vertraue uns, was ist das Schlimmste, das passieren könnte, hmm?)<br/>
1. Clone das Repository<br/>
```bash
git clone https://github.com/thisisjustausername/party_predictor.git
cd party_predictor
```
2. Installiere Dependencies<br/>
```bash
# Install virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

3. Initialisiere die ENV-Variablen<br/>
Recherchiere hierfür den API-Schlüssel der Bundestags-API und ersetze `<Input your API Key here>` im folgenden Code-Snippet damit.
```bash
# Create .env file
touch .env
echo "API_KEY=<Input your API Key here>" >> .env
```

4. Viel Spaß mit dem Code. Beachte, dass alle Dateien mit absoluten Imports geschrieben wurden und daher aus dem Root-Verzeichnis des Repositories gestartet werden müssen.<br/>
```bash
# THIS FILE DOESN'T EXIST. Simply a placeholder for files you want to run.
# Check for correct file path
[[ "$(basename "$PWD")" == "party_predictor" ]] || {echo 'Please run this file from the root directory of the repository.';}

# Run file
python3 -m bert.test_python_file
```

## Training Pipeline
Falls Du das BERT-Modell selbst finetunen willst, verwende diese Pipeline. Wir empfehlen dringend, eine NVIDIA GPU mit mindestens 40GB VRAM und einer Ampere-Architektur zu verwenden.<br/>
1. Lade die Daten herunter und bereite sie vor<br/>
```bash
# Download data
python3 -m dataset_generation.fetch_data

# Preprocess data
python3 -m dataset_generation.clean_data

# OPTIONAL: For the interested ones
python3 -m dataset_generation.investigate_data
```

2. Trainiere das Modell<br/>
Wenn Du willst, kannst Du die Parameter für das Lernen in der Datei `bert/parameters.py` anpassen.
```bash
# Create directories for model stats and finetuned models
mkdir finetuned_model_stats
mkdir finetuned_models

# Train model
python3 -m bert.train
```

3. Evaluiere das Modell<br/>
```bash
python3 -m bert.label_test
```

### Factors
Für die empfohlene Hardware benötigt das Training circa 13 Stunden und konsumiert 4 kWh an Strom.

## Inference

[TODO: finish]: #

## Explain classifications

[TODO: finish]: #

# TODOs
* Verwende eine NER-Maske, um die Klassifizierung nicht von Namen von Personen und Organisationen in den Reden abhängig zu machen
* Verwende random seeds, um das Training deterministisch zu machen.

# Malicious use cases
Das Tool kann für politische Zwecke missbraucht werden. Abgewandelt kann es die Erfolgsfaktoren von bekannten und überzeugenden Reden analysieren und in neuen Reden nachahmen. Dies kann verwendet werden, um populistische und manipulative Reden zu generieren. Es ist anzumerken, dass extreme Reden in den Trainingsdaten enthalten sein können, da diese in diesem Projekt auch analysiert werden können sollen.
