#%%
import shutil
import os
from pathlib import Path
#%%
CLASSES = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet'
]
CLASSES.sort()
#%%
main_di = Path("./data/")
main_tr = main_di/"view-dataset-train/image"
main_te = main_di/"view-dataset-test/image"

main_tr_n = main_di/"view-dataset-train-m10/image"
main_te_n = main_di/"view-dataset-test-m10/image"
# %%
print(main_te_n)
# %%
# os.mkdir(main_te_n); os.mkdir(main_tr_n)

lt = os.listdir(main_tr)

print(len(lt))
new_lt = [x for x in lt if x.split("_")[0] in CLASSES]
new_lts = [x for x in os.listdir(main_te) if x.split("_")[0] in CLASSES]
# print(new_lt[:10])
print(len(new_lt))
[shutil.copy(main_tr/x, main_tr_n/x) for x in new_lt]
[shutil.copy(main_te/x, main_te_n/x) for x in new_lts]