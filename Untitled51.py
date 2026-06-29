#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

# Δεδομένα
t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([69.4, 40.8, 44.4, 27.7, 17.1, 12.3, 7.8, 5.9, 7.2, 3.9])
sy = np.array([13.9, 8.2, 8.9, 5.5, 3.4, 2.5, 1.6, 1.2, 1.4, 0.8])

# Λογάριθμοι και σφάλματα
Y = np.log(y)
sY = sy / y

# Βάρη
w = 1 / sY**2

# Υπολογισμοί για γραμμική παλινδρόμηση με βάρη
S = np.sum(w)
Sx = np.sum(w * t)
Sy = np.sum(w * Y)
Sxx = np.sum(w * t * t)
Sxy = np.sum(w * t * Y)

Delta = S * Sxx - Sx**2

# Παράμετροι
a = (Sxx * Sy - Sx * Sxy) / Delta
b = (S * Sxy - Sx * Sy) / Delta

# Σφάλματα
sigma_a2 = Sxx / Delta
sigma_b2 = S / Delta

# Πίνακας διασποράς
cov_ab = -Sx / Delta
V = np.array([[sigma_a2, cov_ab],
              [cov_ab, sigma_b2]])

# Υπολογισμός y0 και τ
y0 = np.exp(a)
tau = -1 / b

# Σφάλματα
dy0 = y0 * np.sqrt(sigma_a2)
dtau = abs(1 / (b**2)) * np.sqrt(sigma_b2)

# Εκτυπώσεις
print(f"a = {a:.5f}")
print(f"b = {b:.5f}")
print(f"\ny0 = {y0:.3f} ± {dy0:.3f}")
print(f"τ = {tau:.3f} ± {dtau:.3f}")

print("\nΠίνακας διασποράς (V):")
print(V)


# In[ ]:





# In[ ]:





# In[ ]:




