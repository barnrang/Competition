import os
import pandas as pd

data = pd.read_csv('labels.csv')
type_list = pd.read_csv('sample_submission.csv')
all_type = type_list.columns[1:]
os.chdir('train')
for i in all_type:
    os.mkdir(i)

for index, row in data.iterrows():
    os.rename(row['id']+'.jpg',row['breed'] +'/' + str(index)+'.jpg')
os.chdir('..')

for name in all_type:
    os.mkdir(os.path.join('val',name))
for name in all_type:
    dir = os.path.join('train',name)
    file_list = os.listdir(dir)
    for j in range(10):
        os.rename(os.path.join(dir,file_list[j]),'val/'+name+'/'+file_list[j]+'.jpg')
