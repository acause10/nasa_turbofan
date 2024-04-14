
def calculate_RUL(train_data):
    
    unit = max(train_data["UnitNumber"].unique())
    unit_rul = [max(train_data.query('UnitNumber=='+str(i)).Cycle) for i in range(1,unit+1)]
    t_minus_temp = [unit_rul[i-1] - train_data.query('UnitNumber=='+str(i)).Cycle.values for i in range(1,unit+1)]
    t_minus_list = [j for i in t_minus_temp for j in i]
    train_data = train_data.assign(RUL=t_minus_list)
    
    return train_data