import csv
import pandas as pd

def retrieveRow():
    df = pd.read_csv('meta.csv', sep=',')
    row4 = df.loc[df['id'] == "t43"] #Find row with unique id
    nprow4 = row4.to_numpy() #Convert to numpy
    print(nprow4)

def testWriteRow():
    fieldnames = ['id', 'anomalous_pix_perc', 'clear_percent', 'cloud_cover', 'cloud_percent', 'heavy_haze_percent',
                  'origin_x', 'origin_y', 'snow_ice_percent', 'shadow_percent']
    with open(r'meta.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(
            {'id':"t43", 'anomalous_pix_perc': 1, 'clear_percent':  2.0, 'cloud_cover': "test",
             'cloud_percent': 'a', 'heavy_haze_percent': 1, 'origin_x': 2.0,
             'origin_y': 3.0, 'snow_ice_percent': 1.0, 'shadow_percent': 1.0})

def resetCSV():
    with open(r'meta.csv', 'w') as f:
        #Write initial lfields
        f.write('id, anomalous_pix_perc, clear_percent, cloud_cover, cloud_percent, heavy_haze_percent, origin_x, origin_y, snow_ice_percent, shadow_percent\n')

if __name__ == "__main__":
    resetCSV()