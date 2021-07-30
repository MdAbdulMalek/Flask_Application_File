import numpy as np
import re


## Funciton for handling different kinds of size input. This can handle most of the sceanrios.
## But might have to tweak for new cases
def strip_string(s):
    s = str(s)
    if "in" in s.split():
        s1 = s.replace("in", "").rstrip()
    elif "in." in s.split():
        s1 = s.replace("in.", "").rstrip()
    else:
        s1 = s
    tmp = re.split("-| ", s1)
    tmp = list(filter(lambda x: x != "", tmp))
    if len(tmp) == 1 and "/" in tmp[0]:
        tmp_1 = tmp[0].split("/")
        num, denom = tmp_1[0], tmp_1[1]
        return float(num) / float(denom)
    if len(tmp) == 1:
        return float(tmp[0])
    if len(tmp) > 1:
        tmp_1 = float(tmp[0])
        tmp_2 = tmp[1].split("/")
        num, denom = tmp_2[0], tmp_2[1]
        dec = float(num) / float(denom)
        return tmp_1 + dec

    return float(tmp[0])


## Function for finding labels
def find_labels(cat_feat_ind_dict, cat_lab, findings):
    tmp_feat = []
    tmp_lab = []
    tmp_conf = []

    for f in findings:
        lab = f[0]
        conf = f[1]
        label_ind = cat_feat_ind_dict[lab]
        label = cat_lab[label_ind]
        tmp_lab.append(label)
        tmp_conf.append(conf)
        tmp_feat.append(lab)

    return tmp_lab, tmp_conf, tmp_feat
