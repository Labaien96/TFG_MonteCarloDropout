import matplotlib.pyplot as plt
import csv

epoch=[]
acc=[]
loss=[]
val_acc=[]
val_loss=[]
cont=0
with open('VGG16_256_04.csv', 'rt') as csvfile:
    plots= csv.reader(csvfile, delimiter=';')
    for row in plots:
        if cont != 0:
            epoch.append(float(row[0]))
            acc.append(float(row[1]))
            loss.append(float(row[2]))
            val_acc.append(float(row[3]))
            val_loss.append(float(row[4]))
        cont+=1

plt.figure(1)
plt.plot(epoch[1:],acc[1:], label='zehaztasuna entrenamenduan')
plt.plot(epoch[1:],val_acc[1:], label='zehaztasuna balidazioan')
plt.legend()
#plt.title('Data from the CSV File: Accuracy and validation accuracy every epoch')
plt.xlabel('Aro kopurua')
plt.ylabel('Zehaztasuna')

plt.figure(2)
plt.plot(epoch[1:],loss[1:], color='g', label='kostua entrenamenduan')
plt.plot(epoch[1:],val_loss[1:], color='r', label='kostua balidazioan')
plt.legend()
#plt.title('Data from the CSV File: Loss and validation loss every epoch')
plt.xlabel('Aro kopurua')
plt.ylabel('Kostua')
plt.show()
