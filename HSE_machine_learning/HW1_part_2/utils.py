import re

import numpy as np
import pandas as pd


def split_torque(torque):
    '''
    Функция делит столбец torque на два столбца (torque, max_torque_rpm), преобразует
    крутящий момент из 'kgm' в 'Nm' и избавляется от единиц измерения в столбце torque
    '''
    if not pd.isna(torque):
        try:
            if '@' in torque:
                parts = torque.split('@', 1)
                torque = parts[0].strip()
                max_torque_rpm = parts[1].strip()
            elif 'at' in torque:
                parts = torque.split('at', 1)
                torque = parts[0].strip()
                max_torque_rpm = parts[1].strip()

            # Преобразуем `kgm` в `Nm` и избавимся от единиц измерения
            if 'kgm' in torque.lower() or 'kgm' in max_torque_rpm.lower():
                torque = float(re.sub(r'[^\d.]', '', torque)) * 9.8
            elif 'nm' in torque.lower():
                torque = float(re.sub(r'[^\d.]', '', torque))
            else:
                torque = float(torque)
            return torque
        except:
            return np.nan
    else:
        return np.nan
