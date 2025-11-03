import time
import math
import numpy as np  
# NumPy jest używany do szybszych operacji na wektorach i tablicach
# oraz do generowania liczb losowych z różnych rozkładów prawdopodobieństwa


def simulated_annealing(
        func, 
        bounds: list, 
        temp0: float, 
        alpha: float, 
        M: int, 
        k: float, 
        max_iter_total: int
):
    """
    Implementacja algorytmu Symulowanego Wyżarzania (Simulated Annealing, SA).
    Jest to metaheurystyka inspirowana procesem wyżarzania w metalurgii.
    Wersja generyczna, działająca dla N-wymiarowej przestrzeni przeszukiwań.
    
    Parametry:
    :param func: Optymalizowana funkcja (przyjmująca wektor np.array jako argument)
    :param bounds: Lista krotek (min, max) określających granice przeszukiwania dla każdego wymiaru 
    np. [(-105, 105)] dla przestrzeni 1D
    :param temp0: Temperatura początkowa - kontroluje początkową akceptację gorszych rozwiązań
    :param alpha: Współczynnik chłodzenia (0 < alpha < 1) - określa szybkość spadku temperatury
    :param M: Liczba iteracji w każdej temperaturze (pętla wewnętrzna)
    :param k: Stała Boltzmanna w kryterium Metropolisa (zwykle k=1)
    :param max_iter_total: Maksymalna liczba iteracji jako warunek stopu
    
    Zwraca:
    :return: Krotka (najlepsze_rozwiązanie, najlepsza_wartość, czas_wykonania, liczba_iteracji)
    """
    
    # Start pomiaru czasu działania algorytmu
    # Pobieramy znacznik czasu na początku
    start_time = time.time()
    
    # --- 1. Inicjalizacja algorytmu ---
    # Przekształcenie granic na format wymagany przez numpy
    low_bounds = np.array([b[0] for b in bounds])   # Dolne granice dla każdego wymiaru
    high_bounds = np.array([b[1] for b in bounds])  # Górne granice dla każdego wymiaru
    
    # Generowanie losowego punktu startowego z rozkładu jednostajnego w zadanych granicach
    current_solution = np.random.uniform(low=low_bounds, high=high_bounds)
    current_fitness = func(current_solution)  # Obliczenie wartości funkcji celu
    
    # Inicjalizacja najlepszego znalezionego rozwiązania
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # Inicjalizacja zmiennych algorytmu
    act_temp = temp0  # Aktualna temperatura
    total_iter = 0    # Licznik wszystkich wykonanych iteracji

    # Główna pętla algorytmu
    while total_iter < max_iter_total:
        # Pętla wewnętrzna - M prób w stałej temperaturze
        for _ in range(M):
            if total_iter >= max_iter_total:
                break
                
            # --- 2. Generowanie i ocena sąsiedniego rozwiązania ---
            # zastosowano małą modyfikację - przekazujemy act_temp jako step_scale
            # jest to technika z ang. "Adaptive Step Size" lub "Temperature-Dependent Step Size"
            neighbor_solution = generate_neighbor(current_solution, bounds, act_temp)
            neighbor_fitness = func(neighbor_solution)
            
            # --- 3. Ocena i akceptacja rozwiązania ---
            # Obliczenie różnicy wartości (bo szukamy maksimum)
            delta_E = neighbor_fitness - current_fitness
            
            # Przypadek 1: Rozwiązanie lepsze - zawsze akceptowane
            if delta_E > 0:  
                current_solution = neighbor_solution.copy()
                current_fitness = neighbor_fitness
            # Przypadek 2: Rozwiązanie gorsze - akceptacja z prawdopodobieństwem wg kryterium Metropolisa
            else:
                # Prawdopodobieństwo akceptacji maleje wraz ze spadkiem temperatury
                # i wzrostem pogorszenia rozwiązania
                # Rozkład boltzmanna gdy szukamy maksimum
                probability = np.exp(delta_E / (k * act_temp))
                # To właśnie dlatego algorytm jest probabilistyczny
                if np.random.rand() < probability:  # Losowa decyzja o akceptacji
                    # Dlatego też nazywamy ten algorytm "symulowanym wyżarzaniem"
                    current_solution = neighbor_solution.copy()
                    current_fitness = neighbor_fitness
            
            # Aktualizacja najlepszego znalezionego rozwiązania
            if current_fitness > best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
            
            total_iter += 1

        # --- 4. Obniżenie temperatury (schładzanie) ---
        # Nasz "metal" się schładza
        act_temp = act_temp * alpha  # Geometryczne schładzanie 
        # "Geometryczne" bo kolejne wartości temperatury tworzą ciąg geometryczny
        
        # Dodatkowy warunek stopu: jeśli temperatura jest bliska zeru
        # Schemat jest "zamrożony" - algorytm "grzęźnie" 
        # Oznacza to bezpośrednio, że dalsze obliczenia nie mają sensu
        if act_temp < 1e-10:
            break

    # Pomiar czasu po zakończeniu
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Wyświetlenie podsumowania 
    print(f"Zakończono po {total_iter} iteracjach. Czas: {execution_time:.4f} s")
    print(f"Najlepsze rozwiązanie: {best_solution}, Wartość: {best_fitness:.6f}")
    
    # Zwrot wyników
    return best_solution, best_fitness, execution_time, total_iter


def generate_neighbor(
        solution, 
        bounds, 
        step_scale
):
    """
    Generuje sąsiednie rozwiązanie poprzez dodanie losowego zaburzenia.
    Wykorzystuje rozkład normalny (Gaussa) do generowania zaburzeń.
    
    Parametry:
    :param solution: Aktualne rozwiązanie (punkt w przestrzeni)
    :param bounds: Lista krotek (min, max) określających granice dla każdego wymiaru
    :param step_scale: Skala kroku (może być zależna od temperatury)
                      Kontroluje "zasięg" przeskoku do nowego rozwiązania
    
    Zwraca:
    :return: Nowe rozwiązanie (punkt) w dozwolonym obszarze
    """
    # Obliczenie szerokości przedziałów dla każdego wymiaru
    domain_width = np.array([b[1] - b[0] for b in bounds])
    # Dostosowanie wielkości kroku do skali problemu
    step_size = domain_width * step_scale

    # Generowanie zaburzenia z rozkładu normalnego
    # mean=0 - zaburzenia symetryczne wokół obecnego rozwiązania
    # taka sama szansa na pójście "w lewo" jak "w prawo"
    # std=step_size - wielkość zaburzenia zależy od szerokości dziedziny
    neighbor = solution + np.random.normal(0, step_size, size=solution.shape)
    
    # Zabezpieczenie przed wyjściem poza dozwolony obszar
    # np.clip przycina wartości do zadanego przedziału
    low_bounds = np.array([b[0] for b in bounds])
    high_bounds = np.array([b[1] for b in bounds])
    neighbor = np.clip(neighbor, low_bounds, high_bounds)
    
    # Zwracamy propozycję sąsiedniego rozwiązania
    return neighbor