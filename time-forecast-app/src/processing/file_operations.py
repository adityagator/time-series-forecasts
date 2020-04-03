import csv
from processing.constants import Constants
import os
from forecast_project import settings
from django.core.files import File

class FileOperations:

    # def write_forecast_file(data, input):
    #     output_file_name = input.file.name.split("/", 1)
    #     output_file = os.path.join(settings.MEDIA_ROOT ,"output/" + "output_" + output_file_name[1])
    #     with open(output_file, mode='w+') as csv_file:
    #         fieldnames = ['Ship pt', 'Product Hierarchy', 'Part Number', 'Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6',
    #         'Month 7', 'Month 8', 'Month 9', 'Month 10', 'Month 11', 'Month 12', 'Algorithm', 'RMSE', 'MAPE', 'Parameters']
    #         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for key, value in data.items():
    #             ship_pt, prod_h, part_no = key.split("^")
    #             writer.writerow({'Ship pt': ship_pt, 'Product Hierarchy': prod_h, 'Part Number': part_no,
    #             'Month 1': value[3][0], 'Month 2': value[3][1], 'Month 3': value[3][2], 'Month 4': value[3][3], 'Month 5': value[3][4],
    #             'Month 6': value[3][5], 'Month 7': value[3][6], 'Month 8': value[3][7], 'Month 9': value[3][8], 'Month 10': value[3][9],
    #             'Month 11': value[3][10], 'Month 12': value[3][11], 'Algorithm': value[0], 'RMSE': value[1], 'MAPE': value[2], 'Parameters': value[4]})
        
    #     return output_file

    def read_file(file):
        with open(file, mode='r', encoding='utf-8-sig') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            data = list(csv_reader)
        for i in data:
            i[0: 3] = ['^'.join(i[0: 3])]
        dict_data = {i[0]: i[1:] for i in data}

        for key in dict_data:
            nums = dict_data.get(key)
            for i in range(0, len(nums)):
                if(nums[i] == ''):
                    nums[i] = 0
                else:
                    nums[i] = float(nums[i])

        return dict_data

    def write_forecast_file(data, input, vol_cluster, int_cluster):
        output_file_name = input.file.name.split("/", 1)
        output_file = os.path.join(settings.MEDIA_ROOT ,"output/" + "output_" + output_file_name[1])
        with open(output_file, mode='w+') as csv_file:
            fieldnames = ['Ship pt', 'Product Hierarchy', 'Part Number', 'Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6',
            'Month 7', 'Month 8', 'Month 9', 'Month 10', 'Month 11', 'Month 12', 'Algorithm', 'RMSE', 'MAPE', 'Volume', 'Intermittency']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for key, value in data.items():
                ship_pt, prod_h, part_no = key.split("^")
                writer.writerow({'Ship pt': ship_pt, 'Product Hierarchy': prod_h, 'Part Number': part_no,
                'Month 1': value[3][0], 'Month 2': value[3][1], 'Month 3': value[3][2], 'Month 4': value[3][3], 'Month 5': value[3][4],
                'Month 6': value[3][5], 'Month 7': value[3][6], 'Month 8': value[3][7], 'Month 9': value[3][8], 'Month 10': value[3][9],
                'Month 11': value[3][10], 'Month 12': value[3][11], 'Algorithm': value[0], 'RMSE': value[1], 'MAPE': value[2], 'Volume': vol_cluster[key], 'Intermittency': int_cluster[key]})
        
        return output_file