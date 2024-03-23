# %%

file_path = "/data/peiyu/zlc/codebase/layoutcontrol/data/T2I-CompBench_dataset/spatial_train.txt"
with open(file_path, 'r') as f:
    content = [line.strip() for line in f.readlines()]
# %%
import json
content_for_json = []

for idx,prompt in enumerate(content):
    
    content_for_json.append(
        {
            "id":int(idx),
            "prompt":str(prompt)
        }
    )
# %%
save_path = "/data/peiyu/zlc/codebase/layoutcontrol/data/T2I-CompBench_dataset/spatial_train.json"
with open(save_path, 'w') as f:
    json.dump(content_for_json, f)
# %%

test = {'query_id': 0, 'iter': 0, 'prompt': 'a balloon on the top of a giraffe', 'object_list': [['balloon', [0.453125, 0.0, 0.640625, 0.1875]], ['giraffe', [0.375, 0.1875, 0.625, 1.0]]]}

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 画布大小
canvas_size = 512

# 测试数据
# test = {'query_id': 0, 'iter': 0, 'prompt': 'a balloon on the top of a giraffe', 'object_list': [['balloon', [0.453125, 0.0, 0.640625, 0.1875]], ['giraffe', [0.375, 0.1875, 0.625, 1.0]]]}
# test = {'query_id': 0, 'iter': 1, 'prompt': 'a balloon on the top of a giraffe', 'object_list': [['balloon', [0.46875, 0.0, 0.625, 0.15625]], ['giraffe', [0.34375, 0.15625, 0.65625, 1.0]]]}
# test = {'query_id': 2, 'iter': 4, 'prompt': 'a horse on the right of a man', 'object_list': [['horse', [0.484375, 0.15625, 0.984375, 0.8125]], ['man', [0.078125, 0.0625, 0.453125, 0.9375]]]}
# test = {'query_id': 1, 'iter': 4, 'prompt': 'a suitcase on the top of a frog', 'object_list': [['suitcase', [0.234375, 0.078125, 0.78125, 0.515625]], ['frog', [0.234375, 0.546875, 0.78125, 0.984375]]]}
# test = {'query_id': 1, 'iter': 3, 'prompt': 'a suitcase on the top of a frog', 'object_list': [['suitcase', [0.15625, 0.078125, 0.625, 0.46875]], ['frog', [0.078125, 0.46875, 0.703125, 1.0]]]}
# test = {'query_id': 471, 'iter': 2, 'prompt': 'a fish on the right of a airplane', 'object_list': [['fish', [0.703125, 0.421875, 0.921875, 0.578125]], ['airplane', [0.125, 0.359375, 0.515625, 0.546875]]]}
test = {'query_id': 471, 'iter': 1, 'prompt': 'a fish on the right of a airplane', 'object_list': [['fish', [0.625, 0.3125, 0.9375, 0.546875]], ['airplane', [0.0, 0.3125, 0.46875, 0.546875]]]}

# 创建一个新的图形
fig, ax = plt.subplots(1)

# 设置图形的大小
ax.set_xlim([0, canvas_size])
ax.set_ylim([0, canvas_size])

# 遍历每个对象
for obj in test['object_list']:
    # 获取对象的名称和边界框
    name, bbox = obj

    # 将边界框的坐标从相对坐标转换为绝对坐标
    x1, y1, x2, y2 = [coord * canvas_size for coord in bbox]

    # 创建一个新的Rectangle对象
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')

    # 将Rectangle对象添加到图形中
    ax.add_patch(rect)

# 显示图形
plt.show()

# %%
len(content_for_json)
# %%
content_for_json
# %%
import os

def get_filenames_in_dir(directory):
    return os.listdir(directory)

# 使用函数
directory = '/data/peiyu/zlc/codebase/layoutcontrol/data/gligen_generate_images'  # 替换为你的目录路径
filenames = get_filenames_in_dir(directory)
filenames
# %%
len(filenames)
# %%
content_dict = {}
for cont in content_for_json:
    content_dict[cont['id']] = cont['prompt']
content_dict
# %%
import os.path as osp


data_for_finetune = []
for filename in filenames:
    id = filename.split('-')[1].split('_')[0]
    iter = filename.split('_')[1].split('-')[1]
    image_id = filename.split('-')[-1].split('.')[0]
    tmp = {}
    tmp["id"] = id
    tmp["iter"] = iter
    tmp["image_id"] = image_id
    
    image_path = osp.join(directory, filename)
    tmp["image_path"] = image_path
    tmp["prompt"] = content_dict[int(id)]
    
    
    data_for_finetune.append(tmp)

# %%
id,iter,image_id
# %%
data_for_finetune
# %%
save_path = "/data/peiyu/zlc/codebase/layoutcontrol/data/spatial.json"
json.dump(data_for_finetune, open(save_path, 'w'))
# %%
len(data_for_finetune)
# %%
import csv

def read_tsv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        content = list(reader)
    return content

# 使用函数
file_path = '/data/peiyu/zlc/codebase/layoutcontrol/data/DrawBench Prompts - Sheet1.tsv'  # 替换为你的文件路径
content = read_tsv_file(file_path)
print(content)
# %%
prompt_reddit = []
for cont in content:
    if cont[1] == 'Reddit':
        prompt_reddit.append(cont[0])
# %%
len(prompt_reddit)
# %%
with open('/data/peiyu/zlc/codebase/layoutcontrol/data/drawbench_reddit_prompts.txt', 'w') as f:
    for prompt in prompt_reddit:
        f.write(prompt + '\n')  
# %%
with open('/data/peiyu/zlc/codebase/layoutcontrol/data/T2I-CompBench_dataset/spatial_val.txt', "r") as f:
    prompts = [line.strip() for line in f.readlines()]
# %%
prompts
# %%
import json

with open("/data/peiyu/zlc/codebase/layoutcontrol/data/spatial.json", 'r') as f:
    data_for_finetune_only = json.load(f)
    
with open("/data/peiyu/zlc/codebase/layoutcontrol/src/llm_output/spatial/gpt4.spatial.k-similar.k_8.px_64.json", 'r') as f:
    # llm_layout_output = [line.strip() for line in f.readlines()]
    llm_layout_output = json.load(f)
# %%
len(llm_layout_output)
# %%
llm_layout_output[0]
# %%
data_for_finetune_only[2]
# %%
tmp_llm_outout = {}

for item in llm_layout_output:
    if item['query_id'] not in tmp_llm_outout:
        tmp_llm_outout[item['query_id']] = {}
    tmp_llm_outout[item['query_id']][item['iter']] = item
# %%
for item in data_for_finetune_only:
    query_id = int(item['id'])
    iter = int(item['iter'])
    item['object_list'] = tmp_llm_outout[query_id][iter]['object_list']    
# %%
with open("/data/peiyu/zlc/codebase/layoutcontrol/data/spatial_contain_layout.json", 'w') as f:
    json.dump(data_for_finetune_only, f)
# %%
