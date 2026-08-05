# Party Predictor
The European wide trend to right-wing politics hasn't missed Germany. But how clearly is this being reflexted in the Bundestag?<br/>
We went out to analyze whether there is a difference in the speeches of different political orientations. These orientations were determined by the CHES score for right to left wing politics.<br/>
Additionally, we looked at the causes for classifications in order to pinpoint specific choices of words and phrases that are unique to different political orientations and compared them with similar phrases from historical speeches.

# Getting started
Want to use our nice tool but don't want to spend a second enjoying the beautiful code we fabricated? No worries, we got you covered. Simply copy paste all the commands below in your terminal and hit enter every time (trust us, what's the worst that could happen?).<br/>

## Setup
No matter what, run this code!<br/>
1. Clone the repository<br/>
```bash
git clone https://github.com/thisisjustausername/party_predictor.git
cd party_predictor
```
2. Install dependencies<br/>
```bash
# Install virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

3. Initialize the ENV-variables<br/>
For that simply research the API-key of the Bundestag API and replace `<Input your API Key here>` in the following code snippet with it.
```bash
# Create .env file
touch .env
echo "API_KEY=<Input your API Key here>" >> .env
```

4. Have fun running the modules. Note, that all the files are written with absolute imports and should be run from the root directory of the repository as modules.<br/>
```bash
# THIS FILE DOESN'T EXIST. Simply a placeholder for files you want to run.
# Check for correct file path
[[ "$(basename "$PWD")" == "party_predictor" ]] || {echo 'Please run this file from the root directory of the repository.';}

# Run file
python3 -m bert.test_python_file
```

## Training pipeline
In case you want to finetune the BERT-model yourself, use this pipeline. We heavily recommend using a NVIDIA GPU with at least 40GB of VRAM and an Ampere architecture.<br/>
1. Download and preprocess the data<br/>
```bash
# Download data
python3 -m dataset_generation.fetch_data

# Preprocess data
python3 -m dataset_generation.clean_data

# OPTIONAL: For the interested ones
python3 -m dataset_generation.investigate_data
```

2. Train the model<br/>
If you want you can adjust the parameters for learning in the file `bert/parameters.py`.
```bash
# Create directories for model stats and finetuned models
mkdir finetuned_model_stats
mkdir finetuned_models

# Train model
python3 -m bert.train
```

3. Evaluate the model<br/>
```bash
python3 -m bert.label_test
```

### Factors
Training takes around 13 hours and consumes aroung 4 kWh for the mentioned hardware.

## Inference

[TODO: finish]: #

## Explain classifications

[TODO: finish]: #

# TODOs
* Using NER mask the names of persons and organizations in speeches to avoid classifying based on them.
* Make training deterministic using random seeds

# Malicious use cases
Knowing how successful a past speech was this tool can be used to help generate new speeches that are similar in the choice of words and phrases. This can lead to generating populistical and right-extreme speeches (the same applies for left-extreme speeches, too) as such speeches are contained in the training data to allow classifying and analyzing them.
