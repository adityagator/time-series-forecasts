import algorithms as algo
import xlrd
import array as arr
import csv
import pandas as pd


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
with open('input_timeseries.csv',mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data=list(csv_reader)
    
for i in data:
    i[0 : 2] = ['-'.join(i[0 : 2])]
dict_data = {i[0]: i[1:] for i in data}



