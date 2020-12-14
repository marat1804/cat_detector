import numpy as np
import lxml.etree as etree
import random


def glitch_cascade(path):
    with open(cascade_path) as f:
        fxml = f.read()
    root = etree.fromstring(fxml)
    cascade = root.find("cascade")
    list = cascade.getchildren()
    for i in range(len(list)):
        print(i, list[i])

    stages = cascade.find('stages')
    stages_ch = stages.getchildren()

    for i in range(1, len(stages_ch), 2):
        a = stages_ch[i]
        parent = stages_ch[i].getchildren()
        wC = parent[2]
        l = len(wC.getchildren())
        print('before', l)
        for j in range(l - l//3):
            index = random.randint(0, len(wC.getchildren())-1)
            elem = wC.getchildren()[index]
            elem.getparent().remove(elem)
        #th.getparent().remove(th)
        print('after', len(wC.getchildren()))

    tree = root.getroottree()

    tree.write('updated.xml', pretty_print=True)
    with open('updated.xml', 'r') as f:
        a = f.read()

    with open('updated.xml', 'w+') as f:
        f.write('<?xml version="1.0"?>'+'\n')
        f.write(a)





if __name__ == '__main__':
    cascade_path = 'source\cascade_cat.xml'
    glitch_cascade('1.xml')


