#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Άσκηση 5.Το παρακάτω τμήμα κώδικα υπολογίζει τις αθροιστικές κατανομές για τα τυχαία σύνολα τιμών 
#και υπολογίζει τη μέγιστη κατά απόλυτο τιμή απόκλιση D από τις θεωρητικά αναμενόμενες κατανομές. 
#Επίσης αναπαριστά γραφικά τις αθροιστικές κατανομές σε κάθε περίπτωση.


# In[48]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, kstest
# Ρύθμιση για αναπαραγωγιμότητα
n = 10
def plot_ecdf_vs_cdf(sample, dist, dist_name, ax):
    # Ταξινόμηση δείγματος και ECDF
    sample = np.sort(sample)
    ecdf = np.arange(1, n+1) / n
    # Θεωρητική CDF
    x_vals = np.linspace(min(sample) - 1, max(sample) + 1, 1000)
    theoretical_cdf = dist.cdf(x_vals)
    # ECDF για τα ίδια x_vals (με παρεμβολή)
    ecdf_interp = np.searchsorted(sample, x_vals, side='right') / n
    # Υπολογισμός μέγιστης απόκλισης D
    D_statistic, _ = kstest(sample, dist_name)
    D_index = np.argmax(np.abs(ecdf_interp - theoretical_cdf))
    D_x = x_vals[D_index]
    D_y1 = ecdf_interp[D_index]
    D_y2 = theoretical_cdf[D_index]
    
    # Σχεδίαση γραφήματος
    ax.step(sample, ecdf, where='post', color='blue', label='Randomly Generated CDF')
    ax.plot(x_vals, theoretical_cdf, 'orange', linestyle='--', label='Theoretical CDF')
    ax.vlines(D_x, D_y1, D_y2, color='red', linestyle=':', label=f'D={D_statistic:.4f}')
    ax.set_title(f'{dist_name.capitalize()} Distribution (n = {n})')
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Distribution')
    ax.legend()
    ax.grid(True)

    
# Δημιουργία δείγματος
sample_uniform = np.random.uniform(0, 1, n)
sample_normal = np.random.normal(0, 1, n)
# Σχεδίαση δύο διαγραμμάτων
fig, axs = plt.subplots(2, 1, figsize=(6, 10))
plot_ecdf_vs_cdf(sample_uniform, uniform, 'uniform', axs[0])
plot_ecdf_vs_cdf(sample_normal, norm, 'norm', axs[1])
plt.tight_layout()
plt.show()


# In[ ]:


#Εικόνα 1 Αθροιστική κατανομή ομοιόμορφης κατανομής από 10 τυχαίες τιμές και θεωρητική αθροιστική κατανομή. 
#Με κόκκινο φαίνεται η μέγιστη απόσταση D.Εικόνα 2 Αθροιστική κατανομή κανονικής κατανομής 
#από 10 τυχαίες τιμές και θεωρητική αθροιστική κατανομή με κόκκινο πάλι φαίνεται η μέγιστη απόσταση D.


# In[ ]:


#Όπως φαίνεται και παρατηρούμαι από τα παραπάνω γραφήματα τα 10 σημεία δεν είναι αρκετά για να 
# περιγράψουν καλά τη θεωρητική αθροιστική κατανομή και ειδικά στην περίπτωση της ομοιόμορφης 
#κατανομής.Στη συνέχεια, ο κώδικας παράγει 10000 προσπάθειες για κάθε κατανομή και υπολογίζει 
# την ποσότητα √nD= √10D και αναπαράγει το ιστόγραμμα της ποσότητας αυτής.Επίσης, στο κέντρο 
#κάθε bin έχει υπολογιστεί η τιμή της κατανομής Kolmogorov-Smirnov.


# In[49]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, kstwobign

# Ρυθμίσεις
n = 10
num_trials = 10000
bins = np.linspace(0.25, 2.1, 40)  # πιο ρεαλιστικά όρια για sqrt(n)·D

def simulate_distribution(dist_name, dist_generator):
    D_vals = []
    for _ in range(num_trials):
        sample = dist_generator(n)
        D, _ = kstest(sample, dist_name)
        D_vals.append(np.sqrt(n) * D)
    return np.array(D_vals)
def plot_ks_distribution(D_vals, dist_label, ax):
    # Ιστόγραμμα
    hist, bin_edges = np.histogram(D_vals, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Θεωρητική PDF της Kolmogorov–Smirnov (πολλαπλασιασμένη για να ταιριάζει με histogram)
    expected_pdf = kstwobign.pdf(bin_centers)
    expected_scaled = expected_pdf * num_trials * bin_width

    # Σχεδίαση
    ax.hist(D_vals, bins=bins, color='blue', edgecolor='black', label='Observed Kolmogorov Distribution')
    ax.plot(bin_centers, expected_scaled, 'ro', label='Expected Kolmogorov Distribution')

    ax.set_title(f'Kolmogorov - Smirnov from the {dist_label} Distribution (n = {n})', fontsize=12)
    ax.set_xlabel(r'$\sqrt{n} \cdot D$', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend()
          
# Εκτέλεση προσομοιώσεων
D_uniform = simulate_distribution('uniform', lambda n: np.random.uniform(0, 1, n))
D_normal  = simulate_distribution('norm',    lambda n: np.random.normal(0, 1, n))

# Σχεδίαση γραφημάτων
fig, axs = plt.subplots(2, 1, figsize=(7, 10))
plot_ks_distribution(D_uniform, 'Uniform', axs[0])
plot_ks_distribution(D_normal, 'Standard Normal', axs[1])
plt.tight_layout()
plt.show()


# In[ ]:


# Στην εικόνα 3 η κατανομή της ποσότητας √10D για την ομοιόμορφη κατανομή.Στα κέντρα των bins
# εμφανίζεται η θεωρητική τιμή της κατανομής Kolmogorov-Smirnov.Στην εικόνα 4 έχω την 
#Κατανομή της ποσότητας √10D για τις τυχαίες τιμές της κανονικής κατανομής. Στα κέντρα των bins 
#εμφανίζεται η θεωρητική τιμή της κατανομής Kolmogorov-Smirnov.


# In[ ]:


#Οι παρατηρούμενες κατανομές είναι και στις δύο περιπτώσεις μετατοπισμένες προς τα αριστερά σε 
#σχέση με τη θεωρητική. Επομένως, το αναμενόμενο είναι να αποτύχει το test στη συνέχεια θα κάνουμε 
#το χ^2 test εδώ, και οι δύο υποθέσεις θα διατυπώνονται ως εξής για κάθε κατανομή ξεχωριστά:
#δηλαδή,την Μηδενική Υπόθεση H0: Η κατανομή √10D που κατασκευάστηκε ακολουθεί την κατανομή Kolmogorov-Smirnov.
#Εναλλακτική Υπόθεση H1:Η κατανομή √10D δεν ακολουθεί την κατανομή Kolmogorov - Smirnov.


# In[ ]:


#Το παρακάτω τμήμα κώδικα εκτελεί το test για διαφορετικά επίπεδα σημαντικότητας δηλαδή, a=0.01, 
#a=0.05,0.1 και 015. Για n = 10, το test αποδέχεται την H0 για κάθε τιμή του a και για τις δύο κατανομές, 
#το οποίο ήταν και αναμενόμενο, αφού φαινόταν ήδη από τα γραφήματα ότι  υπάρχει συμφωνία 
#μεταξύ των παραγόμενων κατανομών και της κατανομής Kolmogorov-Smirnov.


# In[57]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstwobign, chi2, uniform, norm

# Ρυθμίσεις
n = 10
reps = 10
bins = 10
alpha_levels = [0.01, 0.05, 0.1, 0.15]

def simulate_ks_distribution(n, reps, dist):
    D_vals = []
    for _ in range(reps):
        sample = dist.rvs(size=n)
        sorted_sample = np.sort(sample)
        cdf_vals = dist.cdf(sorted_sample)
        empirical_cdf = np.arange(1, n + 1) / n
        D = np.max(np.abs(empirical_cdf - cdf_vals))
        D_vals.append(np.sqrt(n) * D)
    return np.array(D_vals)

def chi_squared_test_from_histogram(data, bins, dist_pdf):
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    observed = hist / np.sum(hist)

    # Υπολογισμός των αναμενόμενων συχνοτήτων από την K-S κατανομή
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    expected_pdf_vals = dist_pdf(bin_centers)
    expected = expected_pdf_vals / np.sum(expected_pdf_vals)

    mask = expected > 0
    chi_stat = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
    df = np.count_nonzero(mask) - 1
    return chi_stat, df

# Εξομοίωση για τις δύο κατανομές
ks_vals_uniform = simulate_ks_distribution(n, reps, uniform(loc=0, scale=1))
ks_vals_normal = simulate_ks_distribution(n, reps, norm(loc=0, scale=1))
# Έλεγχος Χ² για κάθε επίπεδο σημαντικότητας
for alpha in alpha_levels:
    print(f"\nFor a = {alpha}")

    chi_u, df_u = chi_squared_test_from_histogram(ks_vals_uniform, bins, kstwobign.pdf)
    crit_u = chi2.ppf(1 - alpha, df_u)
    res_u = "REJECTED" if chi_u > crit_u else "ACCEPTED"
    print(f"For the data generated from the uniform distribution, the H0 hypothesis is {res_u}")

    chi_n, df_n = chi_squared_test_from_histogram(ks_vals_normal, bins, kstwobign.pdf)
    crit_n = chi2.ppf(1 - alpha, df_n)
    res_n = "REJECTED" if chi_n > crit_n else "ACCEPTED"
    print(f"For the data generated from the normal distribution, the H0 hypothesis is {res_n}")


# In[ ]:


#Όλα τα παρακάτω βήματα έγιναν και για την τιμή n = 100. Όπως ήταν αναμενόμενο, ο 
#μεγαλύτερος αριθμός τυχαίων τιμών αυξάνει την ακρίβεια στην πρόβλεψη της αθροιστικής κατανομής 
#και τη συμφωνία μεταξύ των κατανομών √100D=10D και της κατανομής Kolmogorov - Smirnov. 
#Παρακάτω παρουσιάζονται τα αντίστοιχα αποτελέσματα.


# In[54]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, kstest

# Ρύθμιση για αναπαραγωγιμότητα
n = 100

def plot_ecdf_vs_cdf(sample, dist, dist_name, ax):
    # Ταξινόμηση δείγματος και ECDF
    sample = np.sort(sample)
    ecdf = np.arange(1, n+1) / n

    # Θεωρητική CDF
    x_vals = np.linspace(min(sample) - 1, max(sample) + 1, 1000)
    theoretical_cdf = dist.cdf(x_vals)

    # ECDF για τα ίδια x_vals (με παρεμβολή)
    ecdf_interp = np.searchsorted(sample, x_vals, side='right') / n

    # Υπολογισμός μέγιστης απόκλισης D
    D_statistic, _ = kstest(sample, dist_name)
    D_index = np.argmax(np.abs(ecdf_interp - theoretical_cdf))
    D_x = x_vals[D_index]
    D_y1 = ecdf_interp[D_index]
    D_y2 = theoretical_cdf[D_index]

    # Σχεδίαση γραφήματος
    ax.step(sample, ecdf, where='post', color='blue', label='Randomly Generated CDF')
    ax.plot(x_vals, theoretical_cdf, 'orange', linestyle='--', label='Theoretical CDF')
    ax.vlines(D_x, D_y1, D_y2, color='red', linestyle=':', label=f'D={D_statistic:.4f}')
    ax.set_title(f'{dist_name.capitalize()} Distribution (n = {n})')
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Distribution')
    ax.legend()
    ax.grid(True)

# Δημιουργία δείγματος
sample_uniform = np.random.uniform(0, 1, n)
sample_normal = np.random.normal(0, 1, n)

# Σχεδίαση δύο διαγραμμάτων
fig, axs = plt.subplots(2, 1, figsize=(6, 10))
plot_ecdf_vs_cdf(sample_uniform, uniform, 'uniform', axs[0])
plot_ecdf_vs_cdf(sample_normal, norm, 'norm', axs[1])

plt.tight_layout()
plt.show()


# In[ ]:


#Εικόνα 5 Αθροιστική κατανομή ομοιόμορφης κατανομής από 100 τυχαίες τιμές και θεωρητική αθροιστική κατανομή. 
#Με κόκκινο φαίνεται η μέγιστη απόσταση D.Εικόνα 6 Αθροιστική κατανομή κανονικής κατανομής από 
#100 τυχαίες τιμές και θεωρητική αθροιστική κατανομή δηλαδή, με κόκκινο φαίνεται η μέγιστη απόσταση D.


# In[55]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, kstwobign

# Ρυθμίσεις
n = 100
num_trials = 10000
bins = np.linspace(0.25, 2.1, 40)  # πιο ρεαλιστικά όρια για sqrt(n)·D

def simulate_distribution(dist_name, dist_generator):
    D_vals = []
    for _ in range(num_trials):
        sample = dist_generator(n)
        D, _ = kstest(sample, dist_name)
        D_vals.append(np.sqrt(n) * D)
    return np.array(D_vals)


def plot_ks_distribution(D_vals, dist_label, ax):
    # Ιστόγραμμα
    hist, bin_edges = np.histogram(D_vals, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Θεωρητική PDF της Kolmogorov–Smirnov (πολλαπλασιασμένη για να ταιριάζει με histogram)
    expected_pdf = kstwobign.pdf(bin_centers)
    expected_scaled = expected_pdf * num_trials * bin_width

    # Σχεδίαση
    ax.hist(D_vals, bins=bins, color='blue', edgecolor='black', label='Observed Kolmogorov Distribution')
    ax.plot(bin_centers, expected_scaled, 'ro', label='Expected Kolmogorov Distribution')

    ax.set_title(f'Kolmogorov - Smirnov from the {dist_label} Distribution (n = {n})', fontsize=12)
    ax.set_xlabel(r'$\sqrt{n} \cdot D$', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend()
        
# Εκτέλεση προσομοιώσεων
D_uniform = simulate_distribution('uniform', lambda n: np.random.uniform(0, 1, n))
D_normal  = simulate_distribution('norm',    lambda n: np.random.normal(0, 1, n))

# Σχεδίαση γραφημάτων
fig, axs = plt.subplots(2, 1, figsize=(7, 10))
plot_ks_distribution(D_uniform, 'Uniform', axs[0])
plot_ks_distribution(D_normal, 'Standard Normal', axs[1])
plt.tight_layout()
plt.show()


# In[ ]:


#Εικόνα 7 Κατανομή της ποσότητας √10D για την ομοιόμορφη κατανομή. Στα κέντρα των bins εμφανίζεται η θεωρητική 
#τιμή της κατανομής Kolmogorov-Smirnov.Εικόνα 8 Κατανομή της ποσότητας √10D για τις τυχαίες τιμές 
#της κανονικής κατανομής. Στα κέντρα των bins εμφανίζεται η θεωρητική τιμή της κατανομής Kolmogorov-Smirnov.


# In[ ]:


#Εδώ η μετατόπιση των παραγόμενων κατανομών προς τα αριστερά είναι λιγότερο εμφανής. 
#Το αυξημένο μέγεθος δείγματος επίσης αλλάζει τελικά και την έκβαση του χ^2 test. 
#Ο αντίστοιχος κώδικας για n = 100 έχει την εξής παρακάτω έξοδο.


# In[56]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstwobign, chi2, uniform, norm

# Ρυθμίσεις
n = 100
reps = 100
bins = 30
alpha_levels = [0.01, 0.05, 0.1, 0.15]

def simulate_ks_distribution(n, reps, dist):
    D_vals = []
    for _ in range(reps):
        sample = dist.rvs(size=n)
        sorted_sample = np.sort(sample)
        cdf_vals = dist.cdf(sorted_sample)
        empirical_cdf = np.arange(1, n + 1) / n
        D = np.max(np.abs(empirical_cdf - cdf_vals))
        D_vals.append(np.sqrt(n) * D)
    return np.array(D_vals)

def chi_squared_test_from_histogram(data, bins, dist_pdf):
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    observed = hist / np.sum(hist)

    # Υπολογισμός των αναμενόμενων συχνοτήτων από την K-S κατανομή
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    expected_pdf_vals = dist_pdf(bin_centers)
    expected = expected_pdf_vals / np.sum(expected_pdf_vals)

    mask = expected > 0
    chi_stat = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
    df = np.count_nonzero(mask) - 1
    return chi_stat, df

# Εξομοίωση για τις δύο κατανομές
ks_vals_uniform = simulate_ks_distribution(n, reps, uniform(loc=0, scale=1))
ks_vals_normal = simulate_ks_distribution(n, reps, norm(loc=0, scale=1))
# Έλεγχος Χ² για κάθε επίπεδο σημαντικότητας
for alpha in alpha_levels:
    print(f"\nFor a = {alpha}")

    chi_u, df_u = chi_squared_test_from_histogram(ks_vals_uniform, bins, kstwobign.pdf)
    crit_u = chi2.ppf(1 - alpha, df_u)
    res_u = "REJECTED" if chi_u > crit_u else "ACCEPTED"
    print(f"For the data generated from the uniform distribution, the H0 hypothesis is {res_u}")

    chi_n, df_n = chi_squared_test_from_histogram(ks_vals_normal, bins, kstwobign.pdf)
    crit_n = chi2.ppf(1 - alpha, df_n)
    res_n = "REJECTED" if chi_n > crit_n else "ACCEPTED"
    print(f"For the data generated from the normal distribution, the H0 hypothesis is {res_n}")


# In[ ]:


#Επομένως, σύμφωνα με το χ2 test, οι κατανομές με n = 100 ακολουθούν την κατανομή Kolmogorov - Smirnov.

