import FileOperations


test = {'USA-A': ['Holt Winters', 10, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
model = FileOperations.FileOperations()
model.write_file(test)
