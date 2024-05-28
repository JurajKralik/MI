import numpy as np

# 1st DeJong function (Sphere)
def dejong1(x):
    return np.sum(x**2)

# 2nd DeJong function (Rosenbrock)
def dejong2(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Schweffel function
def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
def random_search(func, bounds, dim, max_fes):
    best_score = float('inf')
    best_solution = None
    history = []
    
    for _ in range(max_fes):
        solution = np.random.uniform(bounds[0], bounds[1], dim)
        score = func(solution)
        
        if score < best_score:
            best_score = score
            best_solution = solution
        
        history.append(best_score)
    
    return best_solution, best_score, history
def simulated_annealing(func, bounds, dim, max_fes, max_temp, min_temp, cooling_rate, nt):
    best_score = float('inf')
    best_solution = np.random.uniform(bounds[0], bounds[1], dim)
    current_solution = best_solution.copy()
    current_score = func(current_solution)
    history = [current_score]
    
    T = max_temp
    fes = 0
    
    while fes < max_fes and T > min_temp:
        for _ in range(nt):
            neighbor = current_solution + np.random.uniform(-0.1*(bounds[1]-bounds[0]), 0.1*(bounds[1]-bounds[0]), dim)
            neighbor = np.clip(neighbor, bounds[0], bounds[1])
            neighbor_score = func(neighbor)
            
            if neighbor_score < current_score or np.random.rand() < np.exp((current_score - neighbor_score) / T):
                current_solution = neighbor
                current_score = neighbor_score
            
            if current_score < best_score:
                best_score = current_score
                best_solution = current_solution
            
            history.append(best_score)
            fes += 1
            
            if fes >= max_fes:
                break
        
        T *= cooling_rate
    
    return best_solution, best_score, history
import pandas as pd
import matplotlib.pyplot as plt

def run_experiments():
    functions = [dejong1, dejong2, schwefel]
    bounds = [(-5, 5), (-5, 5), (-500, 500)]
    dimensions = [5, 10]
    max_fes = 10000
    runs = 30
    max_temp = 1000
    min_temp = 0.01
    cooling_rate = 0.98
    nt = 10
    
    results = []
    
    for func, bound in zip(functions, bounds):
        for dim in dimensions:
            rs_results = []
            sa_results = []
            
            for _ in range(runs):
                _, rs_score, rs_history = random_search(func, bound, dim, max_fes)
                _, sa_score, sa_history = simulated_annealing(func, bound, dim, max_fes, max_temp, min_temp, cooling_rate, nt)
                
                rs_results.append(rs_score)
                sa_results.append(sa_score)
                
                # Save histories for plotting
                results.append({
                    'function': func.__name__,
                    'dimension': dim,
                    'algorithm': 'RS',
                    'history': rs_history
                })
                results.append({
                    'function': func.__name__,
                    'dimension': dim,
                    'algorithm': 'SA',
                    'history': sa_history
                })
            
            # Statistical analysis
            rs_stats = {
                'Min': np.min(rs_results),
                'Max': np.max(rs_results),
                'Mean': np.mean(rs_results),
                'Median': np.median(rs_results),
                'Std Dev': np.std(rs_results)
            }
            
            sa_stats = {
                'Min': np.min(sa_results),
                'Max': np.max(sa_results),
                'Mean': np.mean(sa_results),
                'Median': np.median(sa_results),
                'Std Dev': np.std(sa_results)
            }
            
            print(f'{func.__name__} (Dimension: {dim}) - Random Search Stats: {rs_stats}')
            print(f'{func.__name__} (Dimension: {dim}) - Simulated Annealing Stats: {sa_stats}')
    
    return results

def plot_results(results):
    for func in ['dejong1', 'dejong2', 'schwefel']:
        for dim in [5, 10]:
            rs_histories = [res['history'] for res in results if res['function'] == func and res['dimension'] == dim and res['algorithm'] == 'RS']
            sa_histories = [res['history'] for res in results if res['function'] == func and res['dimension'] == dim and res['algorithm'] == 'SA']
            
            # Plot individual runs
            plt.figure(figsize=(10, 6))
            for history in rs_histories:
                plt.plot(history, color='blue', alpha=0.3)
            plt.title(f'Random Search Convergence - {func} (Dimension: {dim})')
            plt.xlabel('Iteration')
            plt.ylabel('Best Score')
            plt.show()
            
            plt.figure(figsize=(10, 6))
            for history in sa_histories:
                plt.plot(history, color='red', alpha=0.3)
            plt.title(f'Simulated Annealing Convergence - {func} (Dimension: {dim})')
            plt.xlabel('Iteration')
            plt.ylabel('Best Score')
            plt.show()
            
            # Plot average convergence
            rs_avg_history = np.mean(rs_histories, axis=0)
            sa_avg_history = np.mean(sa_histories, axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(rs_avg_history, color='blue', label='RS')
            plt.plot(sa_avg_history, color='red', label='SA')
            plt.title(f'Average Convergence - {func} (Dimension: {dim})')
            plt.xlabel('Iteration')
            plt.ylabel('Best Score')
            plt.legend()
            plt.show()

results = run_experiments()
plot_results(results)
