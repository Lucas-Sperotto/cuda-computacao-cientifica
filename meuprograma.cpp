#include <iostream>
#include <vector>
#include <lapacke.h>

int main() {
    int n = 3; // Tamanho da matriz
    int lda = n, ldb = n, info;

    // Matriz A (simétrica)
    double A[9] = {4.0, 1.0, 3.0,
                   1.0, 5.0, 2.0,
                   3.0, 2.0, 6.0};

    // Matriz B (simétrica definida positiva)
    double B[9] = {3.0, 1.0, 0.0,
                   1.0, 4.0, 1.0,
                   0.0, 1.0, 5.0};

    std::vector<double> W(n); // Autovalores

    // Chama a rotina LAPACK para resolver o problema de autovalores generalizados
    info = LAPACKE_dsygvd(LAPACK_ROW_MAJOR, 1, 'V', 'U', n, A, lda, B, ldb, W.data());

    if (info == 0) {
        std::cout << "Autovalores:\n";
        for (double lambda : W) {
            std::cout << lambda << " ";
        }
        std::cout << std::endl;
    } else {
        std::cerr << "Erro na chamada da LAPACK, código: " << info << std::endl;
    }

    return 0;
}
