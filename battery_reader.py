'''
 
 dconda install -c anaconda tk
 
'''

import sys
import os
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Tkinter as tk
import datetime 

import matplotlib
matplotlib.use('TkAgg')

import logging 

import os.path

class Device:
    
    def __init__(self,file_path):
        self.name = 'Device_N'
        self.classifyer_names= ['global','channel','statistics']
        self.classifications = {}
        self.data = {}
        self.file_path  = file_path
        self.file_name = os.path.split(file_path)[-1]
        

    def classify_sheets(self,sheet_name):
        ''' '''
        for classifyer in self.classifyer_names:
            if classifyer in sheet_name.lower():
                return classifyer
            
        return 'unknown'
            
    def read_file(self):
        self.excel_file = pd.ExcelFile(self.file_path)
        for sheet_n,sheet_name in enumerate(self.excel_file.sheet_names):
            self.data[sheet_name]  = self.excel_file.parse(sheet_name)
            classification = self.classify_sheets(sheet_name)
            if( classification not in self.classifications.keys() ):
                self.classifications[classification] = []
            self.classifications[classification].append(sheet_name)
            
    def get_name(self):
        if( len(self.classifications['global']) > 0 ):
            sheet_name = self.classifications['global'][0]
            sheet_df = self.data[sheet_name] 
            if( 'TEST REPORT' in sheet_df.keys() and len(sheet_df) > 0 ):
                self.name = sheet_df['TEST REPORT'].iloc[0].split('_')[-1]
            print(sheet_name)
            
            
    def get_mass(self):
        if( len(self.classifications['global']) > 0 ):
            sheet_name = self.classifications['global'][0]
            sheet_df = self.data[sheet_name] 
            if( 'Unnamed: 1' in sheet_df.keys() and len(sheet_df) > 0 ):
                self.mass = sheet_df['Unnamed: 1'].iloc[3].split(':')[-1]
                
            print(sheet_name)
            
    def set_types(self):
        
        classification = 'statistics'
        for sheet_name in  self.classifications[classification]:
            df_i = self.data[sheet_name]
            for col in ['Test_Time(s)','Current(A)','Voltage(V)']:
                if( col in df_i.columns ):
                    df_i[col] =  df_i[col].astype('float')
            
        
        classification = 'channel'
        for sheet_name in  self.classifications[classification]:
            df_i = self.data[sheet_name]
            for col in ['Charge_Capacity(Ah)','Discharge_Capacity(Ah)','True Discharge Capacity']:
                if( col in df_i.columns ):
                    df_i[col] =  df_i[col].astype('float')
            
    def calc_capacity(self):            

        classification = 'statistics'
        for sheet_name in  self.classifications[classification]:
            df_i = self.data[sheet_name]
            df_i['True Charge Capacity'] = df_i['Charge_Capacity(Ah)']*1000.0/self.mass
            df_i['True Discharge Capacity'] = df_i['Discharge_Capacity(Ah)']*1000.0/self.mass
            df_i['Coulumbic Efficiency'] = df_i['True Discharge Capacity']/df_i['True Charge Capacity']
            
    def calc_q(self):
        
        classification = 'statistics'
        for sheet_name in  self.classifications[classification]:
            df_i = self.data[sheet_name]
        
        df_i['Test_Time(h)'] = df_i['Test_Time(s)']/3600.0
        df_i['mA/g'] = df_i['Current(A)']*1000.0/self.mass
        df_i['Q'] = df_i['Test_Time(h)']*abs(df_i['mA/g'])
        
    def calc_dqdv(self,sheet_name,step_index=4,cycle_indx=2):
        df_i = self.data[sheet_name]
        sub_cycle = df_i.loc[ (df_i['Step_Index'] == step_index) & (df_i['Cycle_Index'] == cycle_indx) ]
        # sub_cycle['dQ/dV'] = numerical_der(sub_cycle['Q'].as_matrix(),sub_cycle['Voltage(V)'].as_matrix())
        sub_cycle['dQ/dV'] = np.gradient(channel_df['Q'],channel_df['Voltage(V)'])
                

class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        self.start_time = datetime.datetime.now()
        logging.basicConfig(filename='battery_viewer.log',level=logging.DEBUG)
        logging.info("Start:{}".format(self.start_time.strftime("%y-%m-%d-%H-%M")))
        
        self.resizable(width=True, height=True)
        
        self.minsize(width=666, height=666)
        self.maxsize(width=1666, height=1666)
        
        self.n_entries = 0
        self.entries = []
        self.devices = {}
        #  
        self.add_file_button()
        
        
        
        
    def add_file_button(self):
        '''
        '''
        tk.Label(self, text="File {}".format(self.n_entries)).grid(row=self.n_entries)
        entry = tk.Entry(self)
        entry.grid(row=self.n_entries, column=1)
        self.entries.append(entry)
        
        self.button_file = tk.Button(self, text="+", command=self.add_file_button)
        self.button_file.grid(row=self.n_entries, column=3)
        self.n_entries += 1

        self.button = tk.Button(self, text="Read Data", command=self.read_data)
        self.button.grid(row=self.n_entries, column=1)
        
    
    def read_data(self):
        
        self.data_row = self.n_entries + 2
        #
        for entry_n,entry in enumerate(self.entries):
            
            file_path = entry.get()
            device_i = Device(file_path)

            msg = 'file_path:{}'.format(device_i.file_path) 
            print(msg)
            msg = 'file_name:{}'.format(device_i.file_name) 
            print(msg)
            msg = '{}'.format(device_i.file_name) 
            tk.Label(self, text=msg).grid(row=self.data_row, column=1)
            self.data_row += 1 
            
            if( os.path.exists(file_path) ):
                    
                device_i.read_file()
                
                device_i.get_name()
                
                self.devices[device_i.name] = device_i
                
                for sheet_name in device_i.data.keys():
                    msg = "{}".format(sheet_name)
                    tk.Label(self, text=msg).grid(row=self.data_row, column=1)
                    msg = "X"
                    tk.Label(self, text=msg).grid(row=self.data_row, column=3)
                    self.data_row += 1 
            
            else:
                msg = "File {} not found".format(file_path)
                tk.Label(self, text=msg ).grid(row=self.status_row, column=1)
                self.data_row += 1 
                
            
            

app = SampleApp()
app.mainloop()


'''
def numerical_der(y,x):
    
    .. math::
    
        \frac{dy}{dx}
        
    dyc = np.zeros(len(x))
    dyc[0] = (y[0] - y[1])/(x[0] - x[1])
    for i in range(1,len(y)-1):
        dyc[i] = (y[i+1] - y[i-1])/(x[i+1]-x[i-1])
    dyc[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    # 
    return dyc


global_sheet_name = [sheet_name for sheet_name in xl.sheet_names if 'Global' in sheet_name ][0]
global_df = xl.parse(global_sheet_name)  # read a specific sheet to DataFrame


# In[19]:


device_mass = global_df['Unnamed: 1'].iloc[3].split(':')[-1]


# In[21]:


device_mass = float(device_mass)





# In[15]:


print("device_id:{}".format(device_id))


# In[24]:


statistics_sheet_name = [sheet_name for sheet_name in xl.sheet_names if 'Statistics' in sheet_name ][0]
statistics_df = xl.parse(statistics_sheet_name)  # read a specific sheet to DataFrame


# In[25]:


channel_sheet_name = [sheet_name for sheet_name in xl.sheet_names if 'Channel' in sheet_name ][0]
channel_df = xl.parse(channel_sheet_name)  # read a specific sheet to DataFrame

# In[117]:


# In[60]:


import math


# In[54]:


statistics_df['True Charge Capacity'] = statistics_df['Charge_Capacity(Ah)']*1000.0/device_mass
statistics_df['True Discharge Capacity'] = statistics_df['Discharge_Capacity(Ah)']*1000.0/device_mass
statistics_df['Coulumbic Efficiency'] = statistics_df['True Discharge Capacity']/statistics_df['True Charge Capacity']


# In[119]:


channel_df['Test_Time(h)'] = channel_df['Test_Time(s)']/3600.0


# In[192]:



channel_df['mA/g'] = channel_df['Current(A)']*1000.0/device_mass


# In[205]:


channel_df['Q'] = channel_df['Test_Time(h)']*abs(channel_df['mA/g'])


# In[194]:



# In[131]:


#channel_df['dQ/dV'] = np.gradient(channel_df['Q'],channel_df['Voltage(V)'])


# In[195]:


channel_df['dQ/dV'] = numerical_der(channel_df['Q'].as_matrix(),channel_df['Voltage(V)'].as_matrix())


# In[197]:


sub_cycle


# In[188]:


sub_cycle[['Current(A)','mAh/g','Test_Time(s)','Test_Time(h)','Voltage(V)','Q','dQ/dV']].head()


# In[190]:


(4.022340 - 4.004449)/(223.730220 - 220.390618)


# In[174]:


discharging = channel_df.loc[ channel_df['Cycle_Index'] == 6 ]

# charging['dQ/dV'] = np.gradient(charging['Q'],charging['Voltage(V)'])

discharging = discharging.iloc[1:-2]

plt.plot(discharging['Test_Time(h)'],discharging['dQ/dV'])


# In[55]:


statistics_df = statistics_df.loc[ statistics_df['Cycle_Index'] >= 4 ]


# In[38]:


max_cycles = statistics_df['Cycle_Index'].max()
min_cycles = statistics_df['Cycle_Index'].min()


# In[74]:


min_efficiency = statistics_df['Coulumbic Efficiency'].min()


# In[75]:


min_efficiency


# In[76]:


min_efficiency = math.floor(100.*min_efficiency)/100.0


# In[77]:


min_efficiency


# In[48]:


max_capacity = np.array([statistics_df['True Charge Capacity'].max(),statistics_df['True Charge Capacity'].max()]).max()


# In[49]:


max_capacity_ceil = round(max_capacity,-2)


# In[50]:


max_capacity_int


# In[214]:


# Just a figure and one subplot
f, ax = plt.subplots()

ax.plot(statistics_df['Cycle_Index'],statistics_df['True Charge Capacity'],  label='True Charge Capacity' )
ax.plot(statistics_df['Cycle_Index'],statistics_df['True Discharge Capacity'],label='True Discharge Capacity')

ax.set_title('Capacity vs Cycle')

ax.set_xlabel('Cycle')
ax.set_ylabel('Capacity')

ax.legend(loc="upper right")

ax.set_xlim([0, max_cycles])
ax.set_ylim([0, max_capacity_ceil])

f.subplots_adjust(hspace=0)

f.savefig('{}_Capacity.jpg'.format(device_id))


# In[213]:


# Just a figure and one subplot
f, ax = plt.subplots()

ax.plot(statistics_df['Cycle_Index'],statistics_df['Coulumbic Efficiency'],  label='Coulumbic Efficiency')

ax.set_title('Coulumbic Efficiency')

ax.legend(loc="upper right")

ax.set_xlabel('Cycle')
ax.set_ylabel('Coulumbic Efficiency')

ax.set_xlim([0, max_cycles])
ax.set_ylim([min_efficiency,1.0])

f.subplots_adjust(hspace=0)

f.savefig('{}_Efficiency.jpg'.format(device_id))


# In[215]:


cycle_indx = 2 


f, ax = plt.subplots()


step_index = 4
sub_cycle = channel_df.loc[ (channel_df['Step_Index'] == step_index) & (channel_df['Cycle_Index'] == cycle_indx) ]
sub_cycle['dQ/dV'] = numerical_der(sub_cycle['Q'].as_matrix(),sub_cycle['Voltage(V)'].as_matrix())

ax.plot(sub_cycle['Voltage(V)'],sub_cycle['dQ/dV'], color='b', label='Charge')

step_index = 6
sub_cycle = channel_df.loc[ (channel_df['Step_Index'] == step_index) & (channel_df['Cycle_Index'] == cycle_indx) ]
sub_cycle['dQ/dV'] = numerical_der(sub_cycle['Q'].as_matrix(),sub_cycle['Voltage(V)'].as_matrix())

ax.plot(sub_cycle['Voltage(V)'],sub_cycle['dQ/dV'],  color='b',label='Discharge')


ax.set_title('dQ/dV')

# ax.legend(loc="upper right")

ax.set_xlabel('Voltage(V)')
ax.set_ylabel('dQ/dV (mAh/gV)')

#ax.set_xlim([0, max_cycles])
#ax.set_ylim([min_efficiency,1.0])

f.subplots_adjust(hspace=0)

f.savefig('{}_dQ_dV.jpg'.format(device_id))


# In[ ]:


np.gradient()

'''