import numpy as np
import pandas as pd
import numpy.typing as npt
import scipy.optimize as optimize
class DataGenerator:
    """make predictions from a model"""

    def __init__(self, path: str):
        """initialize class from path to artifact named clf.pickle

        The pickled model must hvae a predict method which accepts
        lists of lists of 4 floats

        """
    @staticmethod
    def generate_noised_lennard_jones_samples(n_samples: int,size: int, seed: int =33) :

        df_good_sample = DataGenerator.generate_lennard_jones_samples(n_samples,size,seed)
        return df_good_sample  
            
    @staticmethod
    def generate_lennard_jones_samples(n_samples: int,size: int, seed: int =33) :

        np.random.seed(seed)
        df_sample = []
        for i in range(n_samples):
            sigma = 3.5*np.random.random()+1.0
            epsilon = 140*np.random.random()+21
            r_0 = sigma * 2**(1/6)
            energy_min = DataGenerator.lennard_jones(sigma, epsilon, r_0)

            # Find Proper boundary: Using bisection (requires a bracketing interval)
            def f_min(r):
                return DataGenerator.lennard_jones(sigma, epsilon, r) - 5*abs(energy_min)
            r_min = optimize.bisect(f_min, 0.01, 100)
            
            def f_max(r):
                return abs(DataGenerator.lennard_jones(sigma, epsilon, r)) - abs(0.05*energy_min)
            r_max = optimize.bisect(f_max, r_min, 100)

            # Generate samples
            df_sample.append(DataGenerator.lennard_jones_pes( sigma, epsilon, r_min, r_max, size))
        
        model_type = ["lennard_jones"]*n_samples    
        return pd.DataFrame({'model_type': model_type,
                             'pes': df_sample,
                             'true_pes': [True]*n_samples})    

    @staticmethod   
    def lennard_jones_pes(
        sigma: float, epsilon: float, R_min: float, R_max: float, size: int
    ) -> pd.DataFrame:
        if size <= 0 or R_min <= 0 or R_max <= 0 or R_min >= R_max:
            raise Exception("Size and range must be positive")

        r = np.linspace(R_min, R_max, size, dtype=np.float64)
        return pd.DataFrame({
            'R': r,
            'Energy': DataGenerator.lennard_jones(sigma, epsilon, r)
        })

    @staticmethod         
    def lennard_jones( sigma: float, epsilon: float, r: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if np.any(r <= 0) :
            raise Exception("Size and range must be positive")

        return  4*epsilon*((sigma/r)**12 - (sigma/r)**6 )
