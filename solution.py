#from scipy.stats import anderson_ksamp
#from hyppo.ksample import MMD
#from scipy.stats import cramervonmises_2samp
#import pandas as pd
#import numpy as np

#chat_id = 653318045 # Ваш chat ID, не меняйте название переменной

#def solution(x: np.array, y: np.array) -> bool:
#    alpha = 0.01
#    
#    model = MMD(
#        compute_kernel='laplacian', bias='True'
#    )
#    _, p_value = model.test(x=x.ravel(), 
#               y=y.ravel()) 
#               
#    
#    return p_value < alpha # Ваш ответ, True или False

from scipy.stats import anderson_ksamp
from hyppo.ksample import Energy
from scipy.stats import cramervonmises_2samp
import pandas as pd
import numpy as np

chat_id = 653318045 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    alpha = 0.01
    
    model = Energy(
        #compute_kernel='laplacian', bias='True'
        compute_distance='canberra', bias='True'
    )
    _, p_value = model.test(x=x.ravel(), 
               y=y.ravel()) 
    
    return p_value < alpha # Ваш ответ, True или False
