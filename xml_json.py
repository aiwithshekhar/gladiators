import cv2
import os
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict
import glob
import json

def xml2csv(xml_path, result):
    """Convert XML to CSV

    Args:
        xml_path (str): Location of annotated XML file
    Returns:
        pd.DataFrame: converted csv file

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
            if result.get(os.path.join('dataset/check',root.find('filename').text)) == None:
                result[os.path.join('dataset/check',root.find('filename').text)]=[value]
            else:
                result.get(os.path.join('dataset/check',root.find('filename').text)).append(value)

    except Exception as e:
        print('xml conversion failed:{}'.format(e))
        return pd.DataFrame(columns=['filename,width,height','class','xmin','ymin','xmax','ymax'])
    return xml_df


xml_files = glob.glob("/home/heh3kor/vaccum_thon/dataset/check/*.xml")
result = defaultdict(list)
for single_xml in xml_files:
    save_info = xml2csv(single_xml, result)

with open('/home/heh3kor/vaccum_thon/dataset/check/check.json', 'w') as outfile:
    json.dump(result, outfile)