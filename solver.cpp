#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <lapacke.h>

using namespace std;
using namespace Eigen;

// Classe para representar um nó na malha
class Node {
public:
    double x, y;
    Node(double x_, double y_) : x(x_), y(y_) {}
};

// Classe para representar um elemento triangular
class Element {
public:
    vector<int> nodeIndices; // Índices dos nós no elemento
    Element(int n1, int n2, int n3) {
        nodeIndices = {n1, n2, n3};
    }
};

// Classe para representar a malha
class Mesh {
public:
    vector<Node> nodes;
    vector<Element> elements;

    void addNode(double x, double y) {
        nodes.emplace_back(x, y);
    }

    void addElement(int n1, int n2, int n3) {
        elements.emplace_back(n1, n2, n3);
    }

    void generateRectangularMesh(double width, double height, int nx, int ny) {
        double dx = width / nx;
        double dy = height / ny;

        for (int j = 0; j <= ny; j++) {
            for (int i = 0; i <= nx; i++) {
                addNode(i * dx, j * dy);
            }
        }

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int n1 = j * (nx + 1) + i;
                int n2 = n1 + 1;
                int n3 = (j + 1) * (nx + 1) + i;
                int n4 = n3 + 1;
                
                addElement(n1, n2, n3);
                addElement(n2, n4, n3);
            }
        }
    }
};

// Classe para montagem das matrizes FEM
class FEMSolver {
public:
    Mesh mesh;
    MatrixXd stiffnessMatrix;
    MatrixXd massMatrix;
    double ar; // Comprimento de referência do guia de onda

    FEMSolver(const Mesh& m, double ar_) : mesh(m), ar(ar_) {
        int size = mesh.nodes.size();
        stiffnessMatrix = MatrixXd::Zero(size, size);
        massMatrix = MatrixXd::Zero(size, size);
    }

    void assembleMatrices() {
        for (const auto& element : mesh.elements) {
            int n1 = element.nodeIndices[0];
            int n2 = element.nodeIndices[1];
            int n3 = element.nodeIndices[2];
            
            double x1 = mesh.nodes[n1].x, y1 = mesh.nodes[n1].y;
            double x2 = mesh.nodes[n2].x, y2 = mesh.nodes[n2].y;
            double x3 = mesh.nodes[n3].x, y3 = mesh.nodes[n3].y;
            
            double area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
            
            Matrix3d localStiffness, localMass;
            localStiffness.setZero();
            localMass.setZero();
            
            Vector3d b, c;
            b << (y2 - y3), (y3 - y1), (y1 - y2);
            c << (x3 - x2), (x1 - x3), (x2 - x1);

            Matrix3d Cx, Cy;

            Cx << (4.0 * b[1]), 0.0, 0.0, (4.0 * b[2]), 0.0, (4.0 * b[3]),
                   0.0, (4.0 * b[2]), 0.0, (4.0 * b[1]), (4.0 * b[3]), 0.0,
                   0.0, 0.0, (4.0 * b[3]), 0.0, (4.0 * b[2]), (4.0 * b[1]),
                   -b[1], -b[2], -b[3], 0.0, 0.0, 0.0;

                   Cy << (4.0 * b[1]), 0.0, 0.0, (4.0 * b[2]), 0.0, (4.0 * b[3]),
                   0.0, (4.0 * b[2]), 0.0, (4.0 * b[1]), (4.0 * b[3]), 0.0,
                   0.0, 0.0, (4.0 * b[3]), 0.0, (4.0 * b[2]), (4.0 * b[1]),
                   -b[1], -b[2], -b[3], 0.0, 0.0, 0.0;


            localMass << 6.0, -1.0, -1.0, 0.0, -4.0, 0.0,
            -1.0, 6.0, -1.0, 0.0, 0.0, -4.0, 
            -1.0, -1.0, 6.0, -4.0, 0.0, 0.0, 
            0, 0, -4.0, 32.0, 16.0, 16.0, 
            -4.0, 0.0, 0.0, 16.0, 32.0, 16.0, 
            0.0, -4.0, 0.0, 16.0, 16.0, 32.0;

            localMass = localMass * (area / 180.0);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    //primeira ordem
                    //localStiffness(i, j) = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area);
                    //localMass(i, j) = (i == j ? 2.0 : 1.0) * area / 12.0;
                 
                 
                    localStiffness(i, j) = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area);
                }
            }
            
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    stiffnessMatrix(element.nodeIndices[i], element.nodeIndices[j]) += localStiffness(i, j);
                    massMatrix(element.nodeIndices[i], element.nodeIndices[j]) += localMass(i, j);
                }
            }
        }
    }

    void solveEigenvalues() {
        int size = mesh.nodes.size();
        int lda = size, ldb = size, info, k = 0;
        
        // Matriz A (simétrica)
        double A[size * size];
    
        // Matriz B (simétrica definida positiva)
        double B[size * size];
    
        std::vector<double> W(size); // Autovalores

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                A[k] = stiffnessMatrix(i, j);
                B[k] = massMatrix(i, j);
                k++;
            }
        }

       
    
        // Chama a rotina LAPACK para resolver o problema de autovalores generalizados
        info = LAPACKE_dsygvd(LAPACK_ROW_MAJOR, 1, 'V', 'U', size, A, lda, B, ldb, W.data());
    
        if (info == 0) {
            std::cout << "Autovalores:\n";
            for (double lambda : W) {
                std::cout << sqrt(lambda) * ar << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Erro na chamada da LAPACK, código: " << info << std::endl;
        }
    



        GeneralizedSelfAdjointEigenSolver<MatrixXd> solver(stiffnessMatrix, massMatrix);
        VectorXd eigenvalues = solver.eigenvalues();

        cout << "Autovalores (k_c * a_r):" << endl;
        for (int i = 0; i < eigenvalues.size(); i++) {
            if (eigenvalues(i) > 0) {
                double kc_ar = sqrt(eigenvalues(i)) * ar;
                cout << kc_ar << endl;
            }
        }
    }
};

int main() {
    double ar = 2.0; // Comprimento de referência do guia de onda
    Mesh mesh;
    mesh.generateRectangularMesh(2.0, 1.0, 20, 10);
    
    FEMSolver solver(mesh, ar);
    solver.assembleMatrices();
    solver.solveEigenvalues();
    
    return 0;
}
