import algorithms as algo
import xlrd
import array as arr

wb = xlrd.open_workbook("test_input.xlsx")
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



