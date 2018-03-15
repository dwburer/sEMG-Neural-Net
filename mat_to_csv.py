import os

import numpy as np
import scipy.io


data_files = [
    'sEMG_Basic_Hand_movements_upatras\\Database 1\\female_1.mat',
    'sEMG_Basic_Hand_movements_upatras\\Database 1\\female_2.mat',
    'sEMG_Basic_Hand_movements_upatras\\Database 1\\female_3.mat',
    'sEMG_Basic_Hand_movements_upatras\\Database 1\\male_1.mat',
    'sEMG_Basic_Hand_movements_upatras\\Database 1\\male_2.mat',
    'sEMG_Basic_Hand_movements_upatras\\Database 2\\male_day_1.mat',
    'sEMG_Basic_Hand_movements_upatras\\Database 2\\male_day_2.mat',
    'sEMG_Basic_Hand_movements_upatras\\Database 2\\male_day_3.mat',
]

for file_path in data_files:
    data = scipy.io.loadmat(file_path)

    for i in data:
        if '__' not in i and 'readme' not in i:
            path_splits = file_path.split('\\')

            new_file_directory = 'sEMG_Basic_Hand_movements_upatras_csv\\%s\\%s' % (path_splits[1], path_splits[2].split('.')[0])
            new_file_path = '%s\\%s.csv' % (new_file_directory, i)

            if not os.path.exists(new_file_directory):
                os.mkdir(new_file_directory)
                
            np.savetxt(new_file_path, data[i], delimiter=',')
