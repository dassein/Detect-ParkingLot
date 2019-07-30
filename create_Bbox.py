import os
import xml.etree.ElementTree as ET
import scipy.io as scio
import numpy as np

def parse_annotation(ann_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}
        tree = ET.parse(ann_dir + ann)
        for elem in tree.iter():
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = []
                for attr in list(elem):
                    if ('name' in attr.tag) & (attr.text == "parking spot"):  # only read in car
                        # obj['name'] = attr.text
                        if attr.text in seen_labels:
                            seen_labels[attr.text] += 1
                        else:
                            seen_labels[attr.text] = 1
                        if len(labels) > 0 and attr.text not in labels:
                            break
                        else:
                            img['object'] += [obj]
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj.append(int(round(float(dim.text))))
                            if 'ymin' in dim.tag:
                                obj.append(int(round(float(dim.text))))
                            if 'xmax' in dim.tag:
                                obj.append(int(round(float(dim.text))))
                            if 'ymax' in dim.tag:
                                obj.append(int(round(float(dim.text))))

        if len(img['object']) > 0:
            all_imgs += [img]
    return all_imgs, seen_labels


def xyminmax2xywh(xyminmax):
    xywh = np.asarray(xyminmax)
    xywh[:, 2:4] = xywh[:, 2:4] - xywh[:, 0:2]
    return xywh.tolist()
if __name__ == "__main__":
    imgs, labels = parse_annotation("./XML/")
    print(labels)
    print(imgs)
    imgs[0]['object'] = xyminmax2xywh(imgs[0]['object'])
    print(imgs[0]['object'])
    scio.savemat('./bbox_parking_spot/Perspective_PL_Obstructed_45.mat', imgs[0])
