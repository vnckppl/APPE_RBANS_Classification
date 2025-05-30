# * Libraries
import dill
import pandas as pd


# * Read RBANS data from user input
print("ENTER YOUR RBANS SCORES:")
print("-----------------------------------------------")
imi_llts = input('Immediate Memory: List Learning raw score:\t')
imi_smts = input('Immediate Memory: Story Memory raw score:\t')
lis_sf   = input('Language: Semantic Fluency raw score:\t\t')
dmi_lrts = input('Delayed Memory: List Recall raw score:\t')
rt_score = input('Delayed Memory: List Recognition raw score:\t')
dmi_srts = input('Delayed Memory: Story Recall raw score:\t')
print("-----------------------------------------------")

# * Import the trained model
import dill
with open('rbans_model.dill.pkl', 'rb') as file:
    predict, predict_proba = dill.load(file)

# * Put the RBANS scores in the correct positional order
rbans_scores = [0, imi_llts, imi_smts] + [0] * 5 + \
    [lis_sf] + [0] * 4 + [dmi_lrts, rt_score, dmi_srts] \
    + [0] * 7


# * Add column labels
import pandas as pd
collabs = ['imi_score', 'imi_llts', 'imi_smts', 'vci_score', 'vci_fcts',
           'vci_lots', 'lis_score', 'lis_pic', 'lis_sf', 'ai_score', 'ai_ds',
           'ai_coding', 'dmi_score', 'dmi_lrts', 'rt_score', 'dmi_srts',
           'dmi_frts', 'tsi_score', 'list_hits', 'list_fp', 'age', 'sex_id',
           'education']
rbans_scores = pd.DataFrame([rbans_scores], columns=collabs)


# * Run the classification and obtain the label
print(
    "\nPREDICTED CLASS:",
    {
        0:"\nCognitively Unimpaired",
        1:"\nMild Cognitive Impairment",
        2:"\nAlzheimer's Disease"
    }[predict(rbans_scores)[0]]
)


# * Obtain probabilistic values
proba = predict_proba(rbans_scores)
print((f"""\nGROUP SPECIFIC CLASSIFICATION PROBABILITY:\n"""
       f"""Cognitively Unimpaired:    {round(proba[0,0],3)}\n"""
       f"""Mild Cognitive Impairment: {round(proba[0,1],3)}\n"""
       f"""Alzheimer's Disease:       {round(proba[0,2],3)}\n"""))
