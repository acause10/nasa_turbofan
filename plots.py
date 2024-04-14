import matplotlib.pyplot as plt
import seaborn as sns

def cycle_frequency(train_data):
    
    max_cycles = []
    units = train_data["UnitNumber"].unique()
    length = len(units)

    for unit in range(1,length+1):
        unit_data = train_data[train_data['UnitNumber']==unit]
        max_cycles.append(max(unit_data['Cycle']))

    plt.hist(x=max_cycles, bins='auto', histtype='bar', rwidth=0.85, color='blue')
    plt.xlabel('Max. number of cycles per unit')
    plt.ylabel('Frequency')
    plt.show()

def sensor_measurements_plot(train_data, unit):

    sensor_measurements = ['Sensor Measure'+str(i) for i in range(1,22)]
    columns = len(sensor_measurements)

    plt.figure(figsize=(17,4*len(sensor_measurements)))
    unit_data = train_data[train_data["UnitNumber"]==unit]
    
    for j in (range(0,columns)):
        a = plt.subplot(columns,1,j+1)
        a.plot(unit_data.index.values,unit_data.iloc[:,j+5].values)
        a.set_title('Sensor Measurement'+str(j+1))
        plt.tight_layout()
    
    plt.show()

def sensor_scatters(train_data):
    
    fig,ax=plt.subplots(7,3,figsize=(40,70))
    x=0
    
    for i in range(0,7):
        for j in range(0,3):
            plt.subplots_adjust(wspace=0.3, hspace=0.7)
            ax[i,j].scatter(train_data['RUL'], train_data['Sensor Measure'+str(x+1)], s=5, alpha=0.30, color='blue')
            ax[i,j].set_title('Sensor Measurement'+str(x+1), fontsize=10)
            ax[i,j].set_xlabel("RUL", fontsize=8)
            ax[i,j].set_ylabel("Sensor Values", fontsize=8)
            ax[i,j].tick_params(axis='both', labelsize=5)
            x += 1
    
    plt.show()

def correlation_matrix(for_extraction):

    corr = for_extraction.corr()
    plt.figure(figsize=(25, 35))
    sns.heatmap(corr,vmax=.6, linewidths=0.01,
            square=True,annot=True,cmap='Blues',linecolor="black")
    plt.title('Correlation between features')
    plt.show()

def plot_prediction(prediction, RUL):

    plt.figure(figsize=(12,5))
    plt.plot((prediction[:2000]), label="Prediction of RUL")
    plt.plot((RUL[:2000]), label="RUL")
    plt.legend()
    plt.show()
