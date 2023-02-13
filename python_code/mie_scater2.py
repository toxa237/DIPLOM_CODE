import numpy as np
import matplotlib.pyplot as plt
from scattnlay import mie


mie.SetLayersSize(np.array([50]))
mie.SetLayersIndex(np.array(1, dtype=np.complex128))
mie.RunMieCalculation()
print(mie.GetFieldE())


