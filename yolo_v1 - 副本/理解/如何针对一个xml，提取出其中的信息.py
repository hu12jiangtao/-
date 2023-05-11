import xml.etree.ElementTree as ET
# 首先是选择对象(树状结构需要用find进行寻找)，之后找出这个对象存储的各个属性值(find)，其中坐标属性之又是一个树状的结构接着用find寻找各个坐标


if __name__ == '__main__':
    path = 'D:\\python\pytorch作业\\计算机视觉\\data\\VOCdevkit\\VOC2007\\Annotations\\000005.xml'
    tree = ET.parse(path)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text # 标签框的名称
        bbox = obj.find('bndbox')
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        obj_struct['bbox'] = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text), int(bbox.find('xmax').text)]  # 标签框的坐标

        objects.append(obj_struct)
    print(objects)
