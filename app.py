import algorithms as algo
import xlrd
import array as arr
import csv
import pandas as pd
import csv


'''wb = xlrd.open_workbook("test_input.xlsx")
sheet = wb.sheet_by_index(0)
i = 2
input_map = {}

while i < sheet.nrows:
    j = 2
    key = sheet.cell_value(i, 0) + "_" + sheet.cell_value(i, 1)
    row_data = []
    while j < sheet.ncols:
        row_data.append(sheet.cell_value(i, j))
        j = j + 1
    input_map[key] = row_data
    i = i + 1

print(input_map)
'''



def formatRawData(raw_file):
    
    with open(raw_file,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data=list(csv_reader)
    for i in data:
        i[0 : 2] = ['-'.join(i[0 : 2])]
    dict_data = {i[0]: i[1:] for i in data}

    for key in dict_data:
        nums = dict_data.get(key)
        for i in range(0,len(nums)):
            nums[i] = int(nums[i])

    return(dict_data)


input_file = 'input_timeseries.csv'
formatted_data = formatRawData(input_file)

for key in formatted_data:
    mov_avg = algorithms.moving_average(formatted_data[key])
    print(mov_avg)





