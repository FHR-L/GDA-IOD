import os
import xml.etree.ElementTree as ET
keep_list_count = {}
remove_list_count = {}

all_class = [
"aeroplane",
"bicycle",
"bird",
"boat",
"bottle",
"bus",
"car",
"cat",
"chair",
"cow",
"diningtable",
"dog",
"horse",
"motorbike",
"person",
"pottedplant",
"sheep",
"sofa",
"train",
"tvmonitor"
]

patter = '5+5'
base = int(patter.split('+')[0])
class_increment = int(patter.split('+')[1])

def write_list_to_file(lst, root):
    with open(root, 'w') as file:
        for item in lst:
            file.write(f"{item}\n")

def remove_person_objects(xml_file_path, output_xml_file_path, keep_list):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    objects_to_remove = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name.lower().strip() not in keep_list:
            objects_to_remove.append(obj)
            if name.lower().strip() in remove_list_count.keys():
                remove_list_count[name.lower().strip()] += 1
            else:
                remove_list_count[name.lower().strip()] = 1
        else:
            if name.lower().strip() in keep_list_count.keys():
                keep_list_count[name.lower().strip()] += 1
            else:
                keep_list_count[name.lower().strip()] = 1

    for obj in objects_to_remove:
        root.remove(obj)

    remaining_objects = root.findall('object')
    if len(remaining_objects) > 0:
        tree.write(output_xml_file_path)
        return True
    else:
        return False

xml_file_path = r'/home/Newdisk/luowenlong/Datasets/VOC2007/Annotations'

def process_directory(input_dir, output_dir, keep_list, train_val_path):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有XML文件
    trainval_list = []
    with open(input_dir, 'r', encoding='utf-8') as file:
        for line in file:
            filename = line.strip()
            input_xml_path = os.path.join(xml_file_path, filename + '.xml')
            output_xml_path = os.path.join(output_dir, filename + '.xml')

            # 处理单个XML文件
            if remove_person_objects(input_xml_path, output_xml_path, keep_list):
                trainval_list.append(filename.split('.')[0].rjust(6,'0'))
    write_list_to_file(trainval_list, train_val_path)

def exma():
    input_dir1 = r'G:\Datasets\PASCOL-VOC\2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\10+10\task1_test.txt'
    input_dir2 = r'G:\Datasets\PASCOL-VOC\2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\ImageSets\Main\test.txt'
    with open(input_dir1, 'r', encoding='utf-8') as file1, open(input_dir2, 'r', encoding='utf-8') as file2:
        for line1, line2 in zip(file1, file2):
            assert line1 == line2
        print("yes")


if __name__ == '__main__':
    # exma()

    input_dir_trainval = r'/home/Newdisk/luowenlong/Datasets/VOC2007/ImageSets/Main/trainval.txt'  # 输入XML文件的目录
    input_dir_test = r'/home/Newdisk/luowenlong/Datasets/VOC2007/ImageSets/Main/test.txt'  # 输入XML文件的目录
    output_dir = r'/home/Newdisk/luowenlong/Datasets/VOC2007/{}/task{}_{}'  # 输出XML文件的目录
    for i in range(int((20-base)/class_increment) + 1):
        keep_list_count = {}
        remove_list_count = {}
        if i == 0:
            keep_list_train_val = all_class[:base]
            keep_list_test = all_class[:base]
            process_directory(input_dir_trainval, output_dir.format(patter, i, 'trainval'), keep_list_train_val,
                              train_val_path=r'/home/Newdisk/luowenlong/Datasets/VOC2007/{}/task{}_trainval.txt'.format(patter, i))
            process_directory(input_dir_test, output_dir.format(patter, i, 'test'), keep_list_test,
                              train_val_path=r'/home/Newdisk/luowenlong/Datasets/VOC2007/{}/task{}_test.txt'.format(
                                  patter, i))
        else:
            keep_list_train_val = all_class[base+(i-1)*class_increment: base+i*class_increment]
            print(keep_list_train_val)
            keep_list_test = all_class[: base+i*class_increment]
            process_directory(input_dir_trainval, output_dir.format(patter, i, 'trainval'), keep_list_train_val,
                              train_val_path=r'/home/Newdisk/luowenlong/Datasets/VOC2007/{}/task{}_trainval.txt'.format(patter, i))
            process_directory(input_dir_test, output_dir.format(patter, i, 'test'), keep_list_test,
                              train_val_path=r'/home/Newdisk/luowenlong/Datasets/VOC2007/{}/task{}_test.txt'.format(
                                  patter, i))
