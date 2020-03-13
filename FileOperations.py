import csv


class FileOperations:

    def write_file(self, data):
        with open('output/output2.csv', mode='w') as csv_file:
            fieldnames = ['location-sku', 'Month 1 - 12 values', 'algorithm', 'rmse', 'mape']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for key, value in data.items():
                writer.writerow({'location-sku': key, 'Month 1 - 12 values': value[3:], 'algorithm': value[0], 'rmse': value[1], 'mape': value[2]})


