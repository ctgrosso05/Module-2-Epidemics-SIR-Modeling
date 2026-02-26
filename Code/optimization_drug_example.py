# drug efficacy optimization example for BME 2315
# made by Lavie, fall 2025

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt


#%% define drug models

# define toxicity levels for each drug (lambda)
metformin_lambda = 0.5

lisinopril_lambda = 0.8

escitalopram_lambda = 0.3

def metformin(x):   # mild toxicity, moderate efficacy
    efficacy = 0.8 * np.exp(-0.1*(x-5)**2)
    toxicity = 0.2 * x**2 / 100
    return efficacy - metformin_lambda * toxicity
def lisinopril(x):  # strong efficacy, higher toxicity
    efficacy = np.exp(-0.1*(x-7)**2)
    toxicity = 0.3 * x**2 / 80
    return efficacy - lisinopril_lambda * toxicity
def escitalopram(x):  # weaker efficacy, low toxicity
    efficacy = 0.6 * np.exp(-0.1*(x-4)**2)
    toxicity = 0.1 * x**2 / 120
    return efficacy - escitalopram_lambda * toxicity
def total_effect(doses):
    return metformin(doses[0]) + lisinopril(doses[1]) + escitalopram(doses[2])   


#%% plot drug efficacies
x = np.linspace(0, 15, 100)
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, metformin(x), label='Metformin', color='blue')
plt.plot(x, lisinopril(x), label='Lisinopril', color='orange')
plt.plot(x, escitalopram(x), label='Escitalopram', color='green')
plt.plot(x, total_effect([x, x, x]), label='Total Effect', color='red', linestyle='--')
plt.title('Drug Efficacy vs Dosage')
plt.xlabel('Dosage (mg)')
plt.ylabel('Net Effect')
plt.legend()
plt.show()

# %% Find optimal dosages for each drug

# First method: Steepest Ascent using the update rule

# first, need the first derivative (gradient)
def gradient(f, x, h=1e-4):
    """Central difference approximation for f'(x) or grad f(x) if x is a vector."""
    x = np.atleast_1d(x)  # Convert to array if scalar
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad.squeeze() if x.size == 1 else grad  # Return scalar if input was scalar

def steepest_ascent(f, x0, h_step=0.1, tol=1e-6, max_iter=1000):
    x = np.atleast_1d(x0)  # Convert to array
    for i in range(max_iter):
        grad = gradient(f, x)
        x_new = x + h_step * grad     
        
        if np.linalg.norm(x_new - x) < tol:  # convergence condition
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    # Convert to Python scalars if single element
    result_x = x.item() if x.size == 1 else x
    result_f = f(x).item() if hasattr(f(x), 'item') and np.asarray(f(x)).size == 1 else f(x)
    return result_x, result_f

# metformin
opt_dose_metformin, opt_effect_metformin = steepest_ascent(metformin, x0=1.0)
print(f"Steepest Ascent Method - Optimal Metformin Dose: {opt_dose_metformin:.2f} mg")
print(f"Steepest Ascent Method - Optimal Metformin Effect: {opt_effect_metformin*100:.2f}%")

# lisinopril
opt_dose_lisinopril, opt_effect_lisinopril = steepest_ascent(lisinopril, x0=1.0)
print(f"Steepest Ascent Method - Optimal Lisinopril Dose: {opt_dose_lisinopril:.2f} mg")
print(f"Steepest Ascent Method - Optimal Lisinopril Effect: {opt_effect_lisinopril*100:.2f}%")

# escitalopram
opt_dose_escitalopram, opt_effect_escitalopram = steepest_ascent(escitalopram, x0=1.0)
print(f"Steepest Ascent Method - Optimal Escitalopram Dose: {opt_dose_escitalopram:.2f} mg")
print(f"Steepest Ascent Method - Optimal Escitalopram Effect: {opt_effect_escitalopram*100:.2f}%")

#total effect
opt_dose_total, opt_effect_total = steepest_ascent(total_effect, x0=np.array([1.0, 1.0, 1.0]))
print(f"Steepest Ascent Method - Optimal Total Doses: {opt_dose_total}")
print(f"Steepest Ascent Method - Optimal Total Effect: {opt_effect_total*100:.2f}%")


# %% Newton's method

# requires second derivative
def second_derivative(f, x, h=1e-4):
    """Central difference approximation for f''(x) or Hessian if x is a vector."""
    x = np.atleast_1d(x)
    n = len(x)
    hess = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            x_pp[i] += h
            x_pp[j] += h
            x_pm[i] += h
            x_pm[j] -= h
            x_mp[i] -= h
            x_mp[j] += h
            x_mm[i] -= h
            x_mm[j] -= h
            hess[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)
    
    return hess.squeeze() if x.size == 1 else hess  # Return scalar if input was scalar

def newtons_method(f, x0, tol=1e-6, max_iter=1000):
    x = np.atleast_1d(x0)
    for i in range(max_iter):
        grad = gradient(f, x)
        hess = second_derivative(f, x)
        
        # Handle scalar vs vector case
        if x.size == 1:
            if hess == 0:  # avoid division by zero
                print("Zero second derivative. No solution found.")
                result_x = x.item() if hasattr(x, 'item') else float(x)
                result_f = f(x).item() if hasattr(f(x), 'item') and np.asarray(f(x)).size == 1 else f(x)
                return result_x, result_f
            x_new = x - grad / hess
        else:
            try:
                x_new = x - np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                print("Singular Hessian. No solution found.")
                result_x = x.item() if x.size == 1 else x
                result_f = f(x).item() if hasattr(f(x), 'item') and np.asarray(f(x)).size == 1 else f(x)
                return result_x, result_f
        
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    # Convert to Python scalars if single element
    result_x = x.item() if x.size == 1 else x
    result_f = f(x).item() if hasattr(f(x), 'item') and np.asarray(f(x)).size == 1 else f(x)
    return result_x, result_f

# metformin
opt_dose_metformin_nm, opt_effect_metformin_nm = newtons_method(metformin, x0=1.0)
print(f"Newton's Method - Optimal Metformin Dose: {opt_dose_metformin_nm:.2f} mg")
print(f"Newton's Method - Optimal Metformin Effect: {opt_effect_metformin_nm*100:.2f}%")                

# lisinopril
opt_dose_lisinopril_nm, opt_effect_lisinopril_nm = newtons_method(lisinopril, x0=1.0)
print(f"Newton's Method - Optimal Lisinopril Dose: {opt_dose_lisinopril_nm:.2f} mg")
print(f"Newton's Method - Optimal Lisinopril Effect: {opt_effect_lisinopril_nm*100:.2f}%")

# escitalopram
opt_dose_escitalopram_nm, opt_effect_escitalopram_nm = newtons_method(escitalopram, x0=1.0)
print(f"Newton's Method - Optimal Escitalopram Dose: {opt_dose_escitalopram_nm:.2f} mg")
print(f"Newton's Method - Optimal Escitalopram Effect: {opt_effect_escitalopram_nm*100:.2f}%")

#total effect
opt_dose_total_nm, opt_effect_total_nm = newtons_method(total_effect, x0=np.array([1.0, 1.0, 1.0]))
print(f"Newton's Method - Optimal Total Doses: {opt_dose_total_nm}")
print(f"Newton's Method - Optimal Total Effect: {opt_effect_total_nm*100:.2f}%")    

#Used Copilot to resolve an error with the Total Effect calculation for the errors. 

# %% Find best lambda value for Metformin using Newton's Method

# Target dose is the metformin dose from the combined effect optimization
target_metformin_dose = opt_dose_total_nm[0]
print(f"Target Metformin Dose (from combined effect): {target_metformin_dose:.2f} mg")

# Test lambda values
lambda_values = np.linspace(0.1, 1.5, 30)
results = []

for test_lambda in lambda_values:
    # Create a temporary metformin function with the test lambda
    def metformin_test(x):
        efficacy = 0.8 * np.exp(-0.1*(x-5)**2)
        toxicity = 0.2 * x**2 / 100
        return efficacy - test_lambda * toxicity
    
    # Find optimal dose using Newton's method
    opt_dose, opt_effect = newtons_method(metformin_test, x0=1.0)
    
    # Calculate the error (difference from target dose)
    error = abs(opt_dose - target_metformin_dose)
    results.append({
        'lambda': test_lambda,
        'optimal_dose': opt_dose,
        'optimal_effect': opt_effect,
        'error': error
    })

# Find the best lambda (smallest error)
best_result = min(results, key=lambda r: r['error'])

print(f"\nBest Lambda Value: {best_result['lambda']:.3f}")
print(f"Optimal Metformin Dose with Best Lambda: {best_result['optimal_dose']:.2f} mg")
print(f"Optimal Metformin Effect with Best Lambda: {best_result['optimal_effect']*100:.2f}%")
print(f"Error from Target Dose: {best_result['error']:.2f} mg")

#Answers to the questions:
#1. A higher lambda value increases the penalty for toxicity, which generally leads to a lower optimal dose. Conversely, a lower lambda value allows for a higher dose since the toxicity penalty is less severe. The optimal dose will vary based on the balance between efficacy and toxicity as determined by the lambda value.
# When increasing lambda, qualitatively, the curve gets steeper and peaks a lot earlier and higher than a lower lambda value. 
#2. Newton's method typically converges faster than the steepest ascent method, especially near the optimum, because it uses second-order information (the Hessian) to adjust the step size and direction. However, it can be more sensitive to the choice of initial guess and may fail to converge if the Hessian is not positive definite or if there are saddle points. Steepest ascent is more robust but can be slower to converge, especially in cases where the function has narrow valleys or plateaus.
#3. It depends. With Newton's method, less iterations has less of an imapct due to how fast it converges. Steepest ascent is a bit slower in its way of getting to the peak, so less iterations would have a drastic effect through this method. 
#4. 
