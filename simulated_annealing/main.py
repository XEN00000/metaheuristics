import simulated_annealing
import demo_functions


if __name__ == "__main__":
    
    # Parametry wejściowe (zakładamy, że takie były w artykule ale można je ustawić inaczej)
    # T0, alpha, M, k
    
    # ----------------------------------------------------
    # --- Funkcja 1 (Rozdział 3) ---
    print("--- Test funkcji z Rozdziału 3 (-2 * |x + 100| + 10) ---")
    
    # Zakładane parametry "z artykułu" do odtworzenia
    PARAMS_CH3 = {
        "func": demo_functions.func_s3,
        "bounds": [(-150.0, 150.0)],    # Dziedzina poszukiwań
        "temp0": 500.0,
        "alpha": 0.997,
        "M": 3000,
        "k": 0.1,
        "max_iter_total": 50000
    }
    simulated_annealing.simulated_annealing(**PARAMS_CH3)
    
    
    # ----------------------------------------------------
    # --- Funkcja 2 (Rozdział 4) ---
    print("\n--- Test funkcji z Rozdziału 4 (x * sin(10 * pi * x) + 1) ---")
    
    # Zakładane parametry "z artykułu" do odtworzenia
    PARAMS_CH4 = {
        "func": demo_functions.func_s4,
        "bounds": [(-1.0, 2.0)],    # Założona dziedzina poszukiwań
        "temp0": 5.0,
        "alpha": 0.997,
        "M": 1200,
        "k": 0.1,
        "max_iter_total": 50000
    }
    simulated_annealing.simulated_annealing(**PARAMS_CH4)