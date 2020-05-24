from sklearn.metrics import auc, roc_curve
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_PATH = './result/GANresult.csv'
df = pd.read_csv(CSV_FILE_PATH,names=['1','2','3','4'])
target = df['1'].values
test1 = df['2'].values
test2 = df['3'].values
test3 = df['4'].values
fpr, tpr, threshold = roc_curve(target, test2)
auc_2 = auc(fpr, tpr)
plt.plot(fpr,tpr,label='f-anoGAN')
print(auc_2)

CSV_FILE_PATH = './result/AEresult.csv'
df = pd.read_csv(CSV_FILE_PATH,names=['1','2'])
target = df['1'].values
test1 = df['2'].values
fpr, tpr, threshold = roc_curve(target, test1)
auc_2 = auc(fpr, tpr)
plt.plot(fpr,tpr,label='AutoEncoder')
print(auc_2)

plt.legend()

plt.savefig('auc.png')
