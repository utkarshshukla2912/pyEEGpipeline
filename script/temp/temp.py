import csv


csvfile1 = open('split_features1.csv', 'w', newline='')
csvfile2 = open('split_features2.csv', 'w', newline='')
writer1 = csv.writer(csvfile1, delimiter=',')
writer2 = csv.writer(csvfile2, delimiter=',')

for i in range(0,10):
    writer1.writerow([1,2,3,4])
    writer2.writerow([1,2,3,4])
