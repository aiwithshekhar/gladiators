import cv2
import os
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import glob
import json
import argparse

def xml2csv(xml_path, result, img_dir):
    """Convert XML to CSV

    Args:
        xml_path (str): Location of annotated XML file
    Returns:
        pd.DataFrame: converted json file

    """
    print("xml to csv {}".format(xml_path))
    xml_list = []
    xml_df=pd.DataFrame()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for member in root.findall('object'):

            value = [int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
            ]
            print (value)
            if result.get(os.path.join(img_dir,root.find('filename').text)) == None:
                result[os.path.join(img_dir,root.find('filename').text)]=[value]
            else:
                result.get(os.path.join(img_dir,root.find('filename').text)).append(value)

    except Exception as e:
        print('xml conversion failed:{}'.format(e))
        return pd.DataFrame(columns=['filename,width,height','class','xmin','ymin','xmax','ymax'])
    return xml_df


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=False,
                        default='/home/heh3kor/vaccum_thon/dataset/Scene_1_selected',
                        help="Path to the sequence.")
    parser.add_argument('--out_json_path', type=str, required=False,
                        default='/home/heh3kor/vaccum_thon/dataset/Scene_2_selected.json',
                        help="Output directory.")

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    img_dir = '/'.join(args.data_dir.split('/')[4:])

    xml_files = glob.glob(f"{args.data_dir}/*.xml")
    result = defaultdict(list)
    for single_xml in xml_files:
        save_info = xml2csv(single_xml, result, img_dir)

    with open(args.out_json_path, 'w') as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    main()