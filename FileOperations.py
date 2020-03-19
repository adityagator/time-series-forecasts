import csv
import Constants


class FileOperations:
    def __init__(self):
        self.constants = Constants.Constants()

    def write_file(self, data):
        with open(self.constants.outputDir + "/" + self.constants.output_file_name, mode='w') as csv_file:
            fieldnames = ['location-sku', 'Month 1 - 12 values', 'algorithm', 'rmse', 'mape']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for key, value in data.items():
                writer.writerow({'location-sku': key, 'Month 1 - 12 values': value[3], 'algorithm': value[0], 'rmse': value[1], 'mape': value[2]})

    def read_file(self):
        with open(self.constants.inputDir + "/" + self.constants.input_file_name, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            data = list(csv_reader)
        for i in data:
            i[0: 3] = ['-'.join(i[0: 3])]
        dict_data = {i[0]: i[1:] for i in data}

        for key in dict_data:
            nums = dict_data.get(key)
            for i in range(0, len(nums)):
                if(nums[i] == ''):
                    nums[i] = 0
                else:
                    nums[i] = int(nums[i])

        return dict_data


