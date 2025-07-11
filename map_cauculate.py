import re

# patterns = ['10+5+5', '15+5', '5+15', '19+1', '10+10']
patterns = ['15+5']
map_list = [
"| aeroplane   | 285  | 918   | 0.870  | 0.701 |",
"| bicycle     | 337  | 2104  | 0.935  | 0.724 |",
"| bird        | 459  | 2253  | 0.878  | 0.569 |",
"| boat        | 263  | 2577  | 0.810  | 0.367 |",
"| bottle      | 469  | 2560  | 0.859  | 0.599 |",
"| bus         | 213  | 2101  | 0.953  | 0.527 |",
"| car         | 1201 | 6210  | 0.954  | 0.807 |",
"| cat         | 358  | 2740  | 0.933  | 0.376 |",
"| chair       | 756  | 8059  | 0.896  | 0.150 |",
"| cow         | 244  | 2850  | 0.963  | 0.515 |",
"| diningtable | 206  | 3623  | 0.932  | 0.577 |",
"| dog         | 489  | 3382  | 0.969  | 0.569 |",
"| horse       | 348  | 1003  | 0.888  | 0.733 |",
"| motorbike   | 325  | 1726  | 0.917  | 0.687 |",
"| person      | 4528 | 17464 | 0.935  | 0.811 |",
"| pottedplant | 480  | 23891 | 0.835  | 0.314 |",
"| sheep       | 242  | 7062  | 0.913  | 0.569 |",
"| sofa        | 239  | 4253  | 0.883  | 0.635 |",
"| train       | 282  | 6155  | 0.901  | 0.489 |",
"| tvmonitor   | 308  | 8607  | 0.860  | 0.573 |",
]

class_to_map = []
for mAP in map_list:
    items = re.split(r'\|', mAP)
    cleaned_items = [item.strip() for item in items if item.strip()]
    class_name = cleaned_items[0]
    class_num = int(cleaned_items[1])
    ap = float(cleaned_items[-1])
    # print(class_name, class_num, ap)
    class_to_map.append((class_name, class_num, ap))

for pattern in patterns:
    parts = pattern.split('+')
    begin = 0
    for task_id, part in enumerate(parts):
        begin_index = begin
        end_index = begin + int(part)
        all_num = 0
        for (class_name, class_num, ap) in class_to_map[begin_index:end_index]:
            all_num += class_num
        mAP = 0
        for (class_name, class_num, ap) in class_to_map[begin_index:end_index]:
            mAP += ap
        mAP = mAP / int(part)
        print("pattern:{}, task{} mAP: {}".format(pattern, task_id, mAP))
        begin += int(part)

    mAP = 0
    for (class_name, class_num, ap) in class_to_map:
        mAP += ap
    mAP = mAP / len(class_to_map)
    print("all mAP: {}".format(mAP))