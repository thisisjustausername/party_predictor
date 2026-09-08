# Party Predictor
*A german version can be found under [Deutsches README](README.de.md).*<br/><br/>
The European wide trend to right-wing politics hasn't missed Germany. For the first time ever the Federal Republic of Germany is shy of getting a Minister-President who is part of a right-extreme party. This trend can be observed in the Bundestag as well with the AfD being the most popular party in current polls. But how clearly is this being reflected by the rethoric used in the Bundestag?<br/>
We went out to analyze whether there is a difference in the speeches of different political orientations. These orientations were determined by the CHES score for right to left wing politics though we performed the classification and uniqueness constraint research independent of this score.<br/>
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
  We recommend using uv as package manager, but pip works just fine as well (maybe a little slower).
  Also do not get the idea to use any python version older than 3.13. The newest used features on it are supported starting at version 3.7, libraries often require higher versions though.
    * uv
      ```bash
      # Install virtual environment
      uv venv --python 3.13 venv
      source venv/bin/activate
      
      # Install dependencies
      uv pip install -r requirements.txt
      ```
    * pip
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

4. Download the BERT-model from Huggingface.<br/>
   Optionally to guarantee a faster download first run `hf auth login` in your terminal to login to Huggingface.
   ```bash
   python3 -m bert.download_bert
   ```

5. Have fun running the modules. Note, that all the files are written with absolute imports and should be run from the root directory of the repository as modules.<br/>
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

## Model results
Training modern-BERt takes around 13 hours and consumes aroung 4 kWh for the mentioned hardware. It results in a F1-score of 0.87.<br/>
The version with Standard-BERT often requires chunking of speeches, though heavily outperforms the modern-BERT approach and also requires far less ressources, already running on a NVIDIA RTX 5000 Mobile (110W) with 16GB VRAM in 8 min/epoch.

The current SOTA model was trained on a train split on 65% of the data (shuffled and randomly selected).<br/>
Hyperparameters were tuned on a validation set that contained 15% of the dataset.<br/>
Finally after training the model was evaluated on a test set making up 20% of the dataset.<br/><br/>
__Parameters__
| Parameter | Value |
| --------- | ----- |
| model | google-bert/bert-base-german-cased |
| context length | 512 tokens |
| batch size | 16 (4 steps in parallel, accumulating over 4 iterations) |
| epochs | 16 |
| learning rate | 3e-5 |
| betas | (0.9,0.999) |
| epsilon | 1e-08 |
| seed | 42 |
| loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Scheduler | LambdaLR |
| Classifier | single linear layer (no activation func) |

<br/>
We achieved following scores on the test data:

__Results__
| Measure | Value |
| ------- | ----- |
| F1-Score | 0.95 |
| Precision | 0.95 |
| Recall | 0.94 |
| Accuracy | 0.94 |

These results are suspiciously good, therefore we want to clarify possible problems that could cause such results.<br/>
We completely avoid data leakage, though implicit information leakage has not yet been handled. For example could a talker mention their colleage or talk positively about their own party. Therefore we need to run NER in order to mask names, organizations, and other information, that gives a hint to the party without carrying any to this project relevant information.<br/>
In general we screened the data for such information leakage and not much has been found. Therefore we assume that this would only have a minor impact.<br/>
Despite that we assume, that the test set is not completely balanced and may produce better results than expected.

[TODO: finish]: #

## Explain classifications

[TODO: finish]: #

# TODOs
* Using NER mask the names of persons and organizations in speeches to avoid classifying based on them.
* Make training deterministic using random seeds
* Update German README
* Avoid information leakage

# Malicious use cases
Knowing how successful a past speech was this tool can be used to help generate new speeches that are similar in the choice of words and phrases. This can lead to generating populistical and right-extreme speeches (the same applies for left-extreme speeches, too) as such speeches are contained in the training data to allow classifying and analyzing them.
