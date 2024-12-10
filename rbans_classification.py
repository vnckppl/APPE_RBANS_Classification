# * Libraries
import dill
import pandas as pd


# * Read RBANS data from user input
print("ENTER YOUR RBANS SCORES:")
print("-----------------------------------")
imi_score = input('Immediate Memory INDEX Score:\t')
imi_llts  = input('List Learning Total Score:\t')
lis_score = input('Language INDEX Score:\t\t')
lis_sf    = input('Semantic Fluency Total Score:\t')
dmi_score = input('Delayed Memory INDEX Score:\t')
dmi_srts  = input('Story Recall Total Score:\t')
print("-----------------------------------")


# * Import the trained model
import dill
with open('rbans_model.dill.pkl', 'rb') as file:
    predict, predict_proba = dill.load(file)


# * Put the RBANS scores in the correct positional order
rbans_scores = [imi_score, imi_llts] + [0] * 4 + \
    [lis_score, 0, lis_sf, 0, 0, 0, dmi_score,
     0, 0, dmi_srts] + [0] * 7


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
