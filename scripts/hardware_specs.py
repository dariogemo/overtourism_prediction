import sys
import pandas as pd

sys.stdout = open('txt_files/metrics_result.txt', 'w')

GPU = 'NVIDIA Tesla P100'
CPU = 'Intel(R) Xeon(R) @ 2.00GHz'
N_CPU = 4

print('\nHARDWARE SPECS AND MODEL METRICS\n',
      '----------------------------------\n'
      f'The models where trained with {N_CPU} {CPU} processors and a single {GPU} as GPU\n')

df = pd.DataFrame([], columns=['Time', 'AVG CPU%', 'AVG GPU%'])
df.loc['Informer'] = [4422, '28.97%', '97.22$']
df.loc['PatchTST'] = [726, '44.90%', '91.46']
df.loc['DLinear'] = [97, '28.97%', '97.22%']

print(df)

sys.stdout.close()
