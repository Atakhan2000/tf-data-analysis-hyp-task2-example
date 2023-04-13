import pandas as pd
import numpy as np
from scipy.stats import anderson_ksamp
from scipy import stats

chat_id = 653318045 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    p_value = anderson_ksamp([x, y]).pvalue 
    #p_value = stats.ks_2samp(x, y)[1]
    #p_value = stats.cramervonmises_2samp(x, y).pvalue
    alpha = 0.01
    return p_value < alpha
