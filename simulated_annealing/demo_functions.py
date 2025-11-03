import math


class section3:
    """
    Klasa zawierająca implementację funkcji z sekcji 3 artykułu.
    """
    def ex1(x):
        """
        Funkcja przykładowa 1 z sekcji 3.
        Definiuje funkcję, która tworzy dwa "wierzchołki" w punktach x=-100 i x=100.
        
        Parametry:
        x (float): Wartość wejściowa x
        
        Zwraca:
        float: Wartość funkcji w punkcie x
        - 10 punktów minus podwójna odległość od x=-100 gdy x jest w przedziale [-105, -95]
        - 10 punktów minus podwójna odległość od x=100 gdy x jest w przedziale [95, 105]
        - 0 dla wszystkich innych wartości x
        """
        # Pierwszy wierzchołek w okolicy x=-100
        if -105 <= x <= -95:
            return -2 * abs(x + 100) + 10
        # Drugi wierzchołek w okolicy x=100
        if 95 < x <= 105:
            return -2 * abs(x + 100) + 10

        # Poza obszarami wierzchołków funkcja zwraca 0
        return 0
    
class section4:
    """
    Klasa zawierająca implementację funkcji z sekcji 4 artykułu.
    """
    def ex4(x):
        """
        Funkcja przykładowa 4 z sekcji 4.
        Implementuje funkcję f(x) = x ⋅ sin(10 ⋅ pi ⋅ x) + 1
        
        Jest to funkcja z wieloma lokalnymi minimami i maksimami ze względu na składnik sinusoidalny.
        
        Parametry:
        x (float): Wartość wejściowa x
        
        Zwraca:
        float: Wartość funkcji w punkcie x według wzoru x * sin(10πx) + 1
        """
        # f(x) = x ⋅ sin(10 ⋅ pi ⋅ x) + 1
        return x * math.sin(10 * math.pi * x) + 1
    
def func_s3(x_vec):
    """
    Funkcja opakowująca (wrapper) dla funkcji z sekcji 3.
    Dostosowuje format wejścia do wymagań algorytmu optymalizacji.
    
    Parametry:
    x_vec (numpy.array): Wektor wejściowy zawierający pojedynczą wartość [x]
    
    Zwraca:
    float: Wynik funkcji section3.ex1 dla pierwszego (i jedynego) elementu wektora
    """
    # x_vec to np.array([x])
    return section3.ex1(x_vec[0])

def func_s4(x_vec):
    """
    Funkcja opakowująca (wrapper) dla funkcji z sekcji 4.
    Dostosowuje format wejścia do wymagań algorytmu optymalizacji.
    
    Parametry:
    x_vec (numpy.array): Wektor wejściowy zawierający pojedynczą wartość [x]
    
    Zwraca:
    float: Wynik funkcji section4.ex4 dla pierwszego (i jedynego) elementu wektora
    """
    # x_vec to np.array([x]) czyli jednowymiarowy wektor/tablica z jedną wartością
    # Robimy to, by pasowało do interfejsu algorytmu optymalizacji
    return section4.ex4(x_vec[0])