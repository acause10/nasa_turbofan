import pandas as pd

def load_train_data():
    
    units_and_cycles = ['UnitNumber','Cycle']
    operational_settings = ['OpSetting1','OpSetting2','OpSetting3']
    sensor_measurements = ['Sensor Measure'+str(i) for i in range(1,22)]
    column_names = units_and_cycles + operational_settings + sensor_measurements

    train_data = pd.read_csv('train_FD001.txt', delim_whitespace=True, header=None)
    train_data.columns = column_names

    return train_data

def load_test_data():

    units_and_cycles = ['UnitNumber','Cycle']
    operational_settings = ['OpSetting1','OpSetting2','OpSetting3']
    sensor_measurements = ['Sensor Measure'+str(i) for i in range(1,22)]
    column_names = units_and_cycles + operational_settings + sensor_measurements

    test_data = pd.read_csv('test_FD001.txt', delim_whitespace=True, header=None)
    test_data.columns = column_names

    return test_data

def load_targets():

    RUL_test = pd.read_table('RUL_FD001.txt',delim_whitespace=True,header=None)

    return RUL_test