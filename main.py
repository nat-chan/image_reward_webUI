# pip install image-reward
import ImageReward as RM
import torch
from pathlib import Path
from tqdm import tqdm
import shutil

model = RM.load("ImageReward-v1.0")
prompt = "Too long chin"
prompt = "square-jawed"
prompt = "big eyes"
prompt = "strabismus"
prompt = "broken eyes"
#prompt = "collapsing"
#prompt = "collapsed face"
prompt2 = prompt.replace(" ", "_")

def single_run(prompt: str, img_path: str) -> float:
    with torch.no_grad():
        reward = model.score(prompt, [img_path])
    return reward

input_dir = f"/mnt/d/v2_1000"
output_dir = f"{input_dir}_{prompt2}"
output_top = f"{input_dir}_{prompt2}_top"
output_bottom = f"{input_dir}_{prompt2}_bottom"
filelist = Path(input_dir).glob("*.png")
print(filelist)

try:
    shutil.rmtree(output_dir)
except:
    pass
Path(output_dir).mkdir(exist_ok=True)

results = list()
for srcpath in tqdm(list(filelist)):
    reward = single_run(prompt, str(srcpath))
    adjusted_reward = reward+10
    print(adjusted_reward, str(srcpath))
    dstpath = Path(output_dir)/f"{adjusted_reward}_{srcpath.stem}"
    results.append((reward, srcpath))
    shutil.copy(srcpath, dstpath)

results.sort(key=lambda x: x[0], reverse=True)

try:
    shutil.rmtree(output_top)
except:
    pass
Path(output_top).mkdir(exist_ok=True)
for score, srcpath in results[:10]:
    dstpath = Path(output_top)/f"{score}_{srcpath.stem}.png"
    shutil.copy(srcpath, dstpath)

try:
    shutil.rmtree(output_bottom)
except:
    pass
Path(output_bottom).mkdir(exist_ok=True)
for score, srcpath in results[-10:]:
    dstpath = Path(output_bottom)/f"{score}_{srcpath.stem}.png"
    shutil.copy(srcpath, dstpath)