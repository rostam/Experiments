////
//// Created by rostam on 14.10.19.
////
//
//#include <iostream>
//#include "Eigen/Dense"
//#include "Eigen/Eigenvalues"
//using namespace Eigen;
//int main()
//{
//    MatrixXd m(2,2);
//    m(0,0) = 3;
//    m(1,0) = 2.5;
//    m(0,1) = -1;
//    m(1,1) = m(1,0) + m(0,1);
//    std::cout << "Here is the matrix m:\n" << m << std::endl;
//    VectorXd v(2);
//    v(0) = 4;
//    v(1) = v(0) - 1;
//    std::cout << "Here is the vector v:\n" << v << std::endl;
//
////    Matrix2d mat;
////    mat << 1, 2,
////            3, 4;
////    Vector2d u(-1,1);
////    Vector2d v(2,0);
////    std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
////    std::cout << "Here is mat*u:\n" << mat*u << std::endl;
////    std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
////    std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
////    std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
////    std::cout << "Let's multiply mat by itself" << std::endl;
////    mat = mat*mat;
////    std::cout << "Now mat is mat:\n" << mat << std::endl;
//
//    MatrixXd ones = MatrixXd::Ones(3,3);
//    VectorXcd eivals = ones.eigenvalues();
//    std::cout << "The eigenvalues of the 3x3 matrix of ones are:" << std::endl << eivals << std::endl;
//}