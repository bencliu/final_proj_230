import csv
import pandas as pd

#Functions for retrieving metadata for a specific itemID
def retrieveRow(id="20181021_162348_102e"):
    df = pd.read_csv('metacopy.csv', sep=',')
    print(df)
    row4 = df.loc[df['id'] == id] #Find row with unique id
    nprow4 = row4.to_numpy()[0] #Convert to numpy
    print(nprow4)

#Function for writing specific row to CSV
def testWriteRow():
    fieldnames = ['id', 'anomalous_pix_perc', 'clear_percent', 'cloud_cover', 'cloud_percent', 'heavy_haze_percent',
                  'origin_x', 'origin_y', 'snow_ice_percent', 'shadow_percent']
    with open(r'metacopy.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(
            {'id':"t43", 'anomalous_pix_perc': 1, 'clear_percent':  2.0, 'cloud_cover': "test",
             'cloud_percent': 'a', 'heavy_haze_percent': 1, 'origin_x': 2.0,
             'origin_y': 3.0, 'snow_ice_percent': 1.0, 'shadow_percent': 1.0})

#Resets CSV with fields
def resetCSV():
    with open(r'metacopy.csv', 'w') as f:
        #Write initial lfields
        f.write('id, anomalous_pix_perc, clear_percent, cloud_cover, cloud_percent, heavy_haze_percent, origin_x, origin_y, snow_ice_percent, shadow_percent\n')

#Change first row of metadatatest.csv file
def changeFirstRow():
    df = pd.read_csv("metacopy.csv")
    print(df.at[0, "anomalous_pix_perc"])
    print(df.at[0, "cloud_cover"])
    print(df.at[0, "origin_x"])
    print(df.at[0, "origin_y"])
    print(df)
    #df.to_csv("metacopy.csv", index=False)

if __name__ == "__main__":
    retrieveRow()