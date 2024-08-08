
from scipy.stats import spearmanr, pearsonr
import numpy as np
import json

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

with open("LLaVA/llave_konvid_val.json","r") as f:
    json_file = json.load(f)

pr_labels = []
gt_labels = []

for i in json_file:
    pr_labels.append(float(i['pred']))
    gt_labels.append(float(i['label']))

pr_labels = rescale(pr_labels, gt_labels)
s = spearmanr(gt_labels, pr_labels)[0]
p = pearsonr(gt_labels, pr_labels)[0]
r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

print("srcc:", s)
print("plcc:", p)
