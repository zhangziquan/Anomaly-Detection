from sklearn.metrics import auc, roc_curve
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_PATH = '/data/weishizheng/x-ray/WGANGP/result/result.csv'
df = pd.read_csv(CSV_FILE_PATH,names=['1','2','3','4'])
target = df['1'].values
test1 = df['2'].values
test2 = df['3'].values
test3 = df['4'].values
fpr, tpr, threshold = roc_curve(target, test1)
auc_1 = auc(fpr, tpr)
plt.plot(fpr,tpr,label='auc:{}'.format(auc_1))
fpr, tpr, threshold = roc_curve(target, test2)
auc_2 = auc(fpr, tpr)
plt.plot(fpr,tpr,label='auc:{}'.format(auc_2))
fpr, tpr, threshold = roc_curve(target, test3)
auc_3 = auc(fpr, tpr)
plt.plot(fpr,tpr,label='auc:{}'.format(auc_3))
print(auc_1)
print(auc_2)
print(auc_3)
plt.switch_backend('agg')
# plt.plot(fpr,tpr,label='auc:{}'.format(auc_1))
plt.savefig('mnist_auc.png')
