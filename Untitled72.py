#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Άσκηση 2 συνέχεια


# In[ ]:


#Στον παρακάτω κώδικα αρχικά κατασκευάζονται τα ιστογράμματα για τις δύο 
#κατανομές καθώς και τα αντίστοιχα σημεία της τυποποιημένης κανονικής κατανομής


# In[112]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Κέντρα των bins
bin_centers = np.array([-2.75, -2.25, -1.75, -1.25, -0.75, -0.25,
                        0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
# Πλάτος bin
bin_width = 0.5
# PDF και σφάλματα
pdf1 = np.array([0.0042, 0.0186, 0.0422, 0.0926, 0.1456, 0.1886,0.1866, 0.1514, 0.0894, 0.0546, 0.0176, 0.0066])
pdf1_err = np.array([0.0009, 0.0019, 0.0029, 0.0043, 0.0054, 0.0061,0.0061, 0.0055, 0.0042, 0.0033, 0.0019, 0.0011])

pdf2 = np.array([0.0056, 0.0214, 0.0422, 0.0844, 0.1296, 0.1660, 0.1736,0.1536, 0.1072, 0.0626, 0.0340, 0.0136])
pdf2_err= np.array([0.0011, 0.0021, 0.0029, 0.0041, 0.0051, 0.0058, 0.0059,0.0055, 0.0046, 0.0035, 0.0026, 0.0016])

theoretical = bin_width * norm.pdf(bin_centers)

plt.figure(figsize=(6, 5))
plt.bar(bin_centers, pdf1, width=0.25, color='blue', label='PDF1', alpha=0.6)
plt.errorbar(bin_centers, pdf1, yerr=pdf1_err, fmt='none', ecolor='black', capsize=4, label='PDF1 Errors')
plt.plot(bin_centers, theoretical, 'ro', label='Theoretical')
plt.title("PDF1 vs Theoretical Normal Distribution")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pdf1_corrected.png", dpi=300)
plt.show()
#Γράφημα 2
plt.figure(figsize=(6, 5))
plt.bar(bin_centers, pdf2, width=0.25, color='blue', label='PDF2', alpha=0.6)
plt.errorbar(bin_centers, pdf2, yerr=pdf2_err, fmt='none', ecolor='black', capsize=4, label='PDF2 Errors')
plt.plot(bin_centers, theoretical, 'ro', label='Theoretical')
plt.title("PDF2 vs Theoretical Normal Distribution")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pdf2_corrected.png", dpi=300)
plt.show()


# In[ ]:


#Στην εικόνα 1 έχω την  Κατανομή PDF1 και τις αναμενόμενες τιμές τις 
# τυποποιημένης κανονικής κατανομής για την PDF1.Στην εικόνα 2 έχω 
#την Κατανομή PDF2  και τις αναμενόμενες τιμές τις τυποποιημένης 
# κανονικής κατανομής για την PDF2.Παρατηρώντας τα δύο γραφήματα
# συμπεραίνουμε ίσως ότι το πιθανότερο είναι η H0 να γίνει αποδεκτή
#στην περίπτωση της PDF1 και να απορριφθεί στην περίπτωση της PDF2.
#Στη συνέχεια ο παρακάτω κώδικας θα υλοποιεί και θα εκτελεί το Pearson’s test.


# In[113]:


import numpy as np
from scipy.stats import norm, chi2 as chi2_dist
from scipy.stats import chi2
# Κέντρα bins
bin_centers = np.array([
    -2.75, -2.25, -1.75, -1.25, -0.75, -0.25,
     0.25,  0.75,  1.25,  1.75,  2.25,  2.75])
# PDF1 και PDF2 (παρατηρούμενες τιμές)
pdf1_dx = np.array([
    0.0042, 0.0186, 0.0422, 0.0926, 0.1456, 0.1886,
    0.1866, 0.1514, 0.0894, 0.0546, 0.0176, 0.0066
])
pdf2_dx = np.array([
    0.0056, 0.0214, 0.0422, 0.0844, 0.1296, 0.1660,
    0.1736, 0.1536, 0.1072, 0.0626, 0.0340, 0.0136])

# Σφάλματα (το ± σε κάθε PDF)
err1 = np.array([
    0.0009, 0.0019, 0.0029, 0.0043, 0.0054, 0.0061,
    0.0061, 0.0055, 0.0042, 0.0033, 0.0019, 0.0011
])

err2 = np.array([
    0.0011, 0.0021, 0.0029, 0.0041, 0.0051, 0.0058,
    0.0059, 0.0055, 0.0046, 0.0035, 0.0026, 0.0016
])
# Πλάτος bin και Υπολογισμός θεωρητικής πυκνότητας N(0,1) σε κάθε bin (ακέραια PDF * dx)
bin_width = 0.5
theoretical = norm.pdf(bin_centers, loc=0, scale=1) * bin_width
# Υπολογισμός χ² για κάθε PDF
chi2_pdf1 = np.sum(((pdf1_dx - theoretical)**2) / (err1**2))
chi2_pdf2 = np.sum(((pdf2_dx - theoretical)**2) / (err2**2))

# Κριτική τιμή για α = 0.1 και dof = 12
alpha = 0.1
dof = 12
critical_value = chi2_dist.ppf(1 - alpha, dof)
# Εκτύπωση αποτελεσμάτων
print(f"Για το τεστ chi^2 του Pearson:")
print(f"Η κρίσιμη τιμή για το chi^2 σε α = {alpha} είναι: {critical:.3f}")
print(f"Το τεστ chi^2 του Pearson για το PDF1 είναι: {chi2_pdf1:.3f}")
print(f"Η υπόθεση H0 {'ΑΠΟΡΡΙΠΤΕΤΑΙ' if chi2_pdf1 > critical else 'ΔΕΝ ΑΠΟΡΡΙΠΤΕΤΑΙ'} για το PDF1")
print(f"Το τεστ chi^2 του Pearson για το PDF2 είναι: {chi2_pdf2:.3f}")
print(f"Η υπόθεση H0 {'ΑΠΟΡΡΙΠΤΕΤΑΙ' if chi2_pdf2 > critical else 'ΔΕΝ ΑΠΟΡΡΙΠΤΕΤΑΙ'} για το PDF2")

theoretical = expected
obs = pdf1_dx 
obs = pdf2_dx 
def pearson_chi_squared(obs, exp, err, dof):
    chi2_stat = np.sum((obs - exp)**2 / err**2)
    p_value = chi2.sf(chi2_stat, dof)
    return chi2_stat, p_value
dof = 12
chi1, p1_value = pearson_chi_squared(obs1, expected, err1, dof)
chi2, p2_value = pearson_chi_squared(obs2, expected, err2, dof)
print("\n")
print(chi1,'και', p1_value)
print(chi2,'και', p2_value)


# In[ ]:


#Βλέπουμε ότι όπως ήταν αναμενόμενο, το test απορρίπτει την H0 για την PDF2, ωστόσο την 
#απορρίπτει επίσης και για την PDF1 πράγμα που δεν το περιμέναμε με βάση το γράφημα.
#Παρακάτω έχουμε τον κώδικα για το RUN test.


# In[118]:


import numpy as np
import scipy.stats as stats

# Δεδομένα
pdf1 = np.array([0.0042, 0.0186, 0.0422, 0.0926, 0.1456, 0.1886,
                 0.1866, 0.1514, 0.0894, 0.0546, 0.0176, 0.0066])
pdf2 = np.array([0.0056, 0.0214, 0.0422, 0.0844, 0.1296, 0.1660,
                 0.1736, 0.1536, 0.1072, 0.0626, 0.0340, 0.0136])

# Ορισμός bins και κέντρων
bins = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
                 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
bin_centers = (bins[:-1] + bins[1:]) / 2
dx = 0.5

# Υπολογισμός θεωρητικής PDF (τυποποιημένη κανονική)
theoretical_pdf = stats.norm.pdf(bin_centers) * dx

# Συνάρτηση Run Test
def run_test(diff):
    signs = np.sign(diff)
    signs = signs[signs != 0]
    N_pos = np.sum(signs > 0)
    N_neg = np.sum(signs < 0)
    r = 1 + np.sum(signs[1:] != signs[:-1]) if len(signs) > 1 else 0
    N = N_pos + N_neg
    if N < 2:
        return r, N_pos, N_neg, None, None, None, None
    E_r = 1 + (2 * N_pos * N_neg) / N
    V_r = (2 * N_pos * N_neg * (2 * N_pos * N_neg - N_pos - N_neg)) / ((N ** 2) * (N - 1))
    z = (r - E_r) / np.sqrt(V_r)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return r, N_pos, N_neg, E_r, V_r, z, p_value

# Υπολογισμός διαφορών και εφαρμογή Run Test
diff1 = pdf1 - theoretical_pdf
diff2 = pdf2 - theoretical_pdf

results_pdf1 = run_test(diff1)
results_pdf2 = run_test(diff2)

# Εκτύπωση αποτελεσμάτων
def display_results(label, results):
    r, Np, Nn, E_r, V_r, z, p = results
    print(f"{label}:")
    print(f"z = {z} με αντίστοιχη τιμή p value of: {p}")
    print("Η Η0 υπόθεση ", "είναι ΑΠΟΔΕΚΤΗ" if p > 0.1 else "ΑΠΟΡΡΗΦΘΗΚΕ", f"for {label}")
    print()

print("Οι υπόλοιπες ποσότητες πουχρησιμοποιήθηκαν για τον υπολογισμό των z δίνονται στον παρακάτω πίνακα:")
print(f"{'':<10}{'PDF1':>10}{'PDF2':>10}")
print(f"{'r':<10}{results_pdf1[0]:>10}{results_pdf2[0]:>10}")
print(f"{'N+':<10}{results_pdf1[1]:>10}{results_pdf2[1]:>10}")
print(f"{'N-':<10}{results_pdf1[2]:>10}{results_pdf2[2]:>10}")
print(f"{'E(r)':<10}{round(results_pdf1[3], 3):>10}{round(results_pdf2[3], 3):>10}")
print(f"{'V(r)':<10}{round(results_pdf1[4], 3):>10}{round(results_pdf2[4], 3):>10}")

# Τελική παρουσίαση z και p-value
display_results("PDF1", results_pdf1)
display_results("PDF2", results_pdf2)


# In[ ]:


#Στον παραπάνα πίνακα βλέπουμε διάφορες ποσότητες που χρησιμοποιήθηκαν και προκύπτουν 
#από το RUN test για τα δύο σύνολα δεδομένων. Για το RUN test τα αποτελέσματα
#ακολουθούν αυτό που αναφέρθηκε παραπάνω όταν κάναμε τα γραφήματα δηλαδή,ότι 
#αποδέχεται την H0 για την PDF1 ενώ,την απορρίπτει για την PDF2. Ωστόσο, 
#η απόρριψη της H0 για την PDF1 από το Pearson’s test δημιουργεί προβληματισμούς. 
#Για αυτόν τον λόγο θα πρέπει να χρησιμοποιήσουμε και ένα ακόμη test και 
#έτσι προτιμάται να χρησιμοποιηθεί ένα συνδυαστικό test Fisher, όπως και θα δούμε παρακάτω.


# In[ ]:


#Το Fisher test χρησιμοποιείται για να συνδυαστούν n ανεξάρτητα tests τα οποίαεξετάζουν 
# την ίδια μηδενική υπόθεση H0. Εάν pi είναι η p value για το i−οστό test,τότε μπορεί 
# να υπολογιστεί μία συνδυασμένη p value μέσω της παρακάτω ποσότητας:
# α= -2*ln(Πi=1,n pi)= -2* Σi=1,n(lnpi) και η οποία ακολουθεί την κατανομή χ^2 
#με 2*n βαθμούς ελευθερίας.Τότε η νέα p value  θα συγκρίνεται με το α και
#εάν είναι μικρότερη τότε θα  απορρίπτεται η H0. Στον παρακάτω κώδικα 
#θα υλοποιείσω και θα εκτελέσω το Fisher test.


# In[119]:


import numpy as np
from scipy.stats import chi2
# Δοσμένες τιμές p value από Pearson και RUN test 
p1_pearson = 0.04123773549736348
p2_pearson = 4.236483306104561e-33
p1_run = 0.5448268505172666
p2_run = 0.016648970490260654
def fisher(p_values):
    n = len(p_values)
    chi_stat = -2 * np.sum(np.log(p_values))
    combined_p = chi2.sf(chi_stat, 2 * n)
    return combined_p
# Υπολογισμός για κάθε PDF
p1_combined = fisher([p1_pearson, p1_run])
p2_combined = fisher([p2_pearson, p2_run])

# Εκτύπωση αποτελεσμάτων
print("Combining the two tests:")
print(f"The combined p value for PDF1 is: {p1_combined}")
if p1_combined > 0.1:
    print("The H0 hypothesis is ACCEPTED for PDF1")
else:
    print("The H0 hypothesis is REJECTED for PDF1")
print(f"The combined p value for PDF2 is: {p2_combined}")
if p2_combined > 0.1:
    print("The H0 hypothesis is ACCEPTED for PDF2")
else:
    print("The H0 hypothesis is REJECTED for PDF2")


# In[ ]:


#Τελικά,το συνδυαστικό test Fisher επιβεβαιώνει τις προβλέψεις που έγιναν αρχικά μέσω
#των γραφημάτων δηλαδή, η H0 γίνεται αποδεκτή για την κατανομή PDF1 και απορρίπτεται
#για την PDF2.∆ηλαδή η PDF1 ακολουθεί τυποποιημένη κανονική κατανομή ενώ η PDF2 όχι.

