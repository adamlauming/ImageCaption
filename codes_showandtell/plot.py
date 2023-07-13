import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

train_loss_list = []
val_loss_list = []
bleu_list = []


with open('./save_model/20210309/logs.txt', 'r') as f:
    losses = f.readlines()

x1 = range(0, len(losses))
x2 = range(0, len(losses))
x3 = range(0, len(losses))
for line in losses[:100]:
    train_loss_list.append(float(line.strip().split()[0]))
    val_loss_list.append(float(line.strip().split()[1]))
    bleu_list.append(float(line.strip().split()[2]))


y1 = train_loss_list
y2 = val_loss_list
y3 = bleu_list

plt.subplot(3, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('train_loss vs. epoches')
plt.ylabel('train_loss')

plt.subplot(3, 1, 2)
plt.plot(x2, y2, '*-')
plt.title('val_loss vs. epoches')
plt.ylabel('val_loss')

plt.subplot(3, 1, 3)
plt.plot(x3, y3, '*-')
plt.title('bleu vs. epoches')
plt.ylabel('bleu')

plt.savefig("figure.jpg")

plt.show()































