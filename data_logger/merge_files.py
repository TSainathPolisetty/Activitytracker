import csv

def append_csv(input_filename, output_filename):
    with open(output_filename, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        
        with open(input_filename, 'r') as infile:
            reader = csv.reader(infile)
            next(reader)            
            for row in reader:
                writer.writerow(row)

input_file = "Data/idle_data.csv"
output_file = "Data/ex_data.csv"
append_csv(input_file, output_file)
