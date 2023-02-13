#include <iostream>
#include <complex>
#include <vector>


const double PI = 3.141592653589793238463;

class Mie_field
{
private:
    std::vector<std::complex<double>> a_;
    std::vector<std::complex<double>> b_;
    std::vector<std::complex<double>> c_;
    std::vector<std::complex<double>> d_;
    // std::vector<std::complex<double>> a (2, std::complex<double>(0, 0));

public:
    double R;
    double K;
    double N1;
    double N2;
    int E0;
    double STEP;
    int SIZE ;
    double count_harmonics = 3;

    Mie_field(int lambda, int r, double step, double eps_1, int size, int e0);
    Mie_field();

    void print(){
        std::cout << "R " << R << std::endl;
        std::cout << "K " << K << std::endl;
        std::cout << "N1 " << N1 << std::endl;
        std::cout << "N2 " << N2 << std::endl;
        std::cout << "E0 " << E0 << std::endl;
        std::cout << "STEP " << STEP << std::endl;
        std::cout << "SIZE " << SIZE << std::endl;
        std::cout << "count_harmonics " << count_harmonics << std::endl;
        // for(int i=1; i<SIZE; i++){
        //     std::cout << XYZ[i] << std::endl;
        // }
    }
};


Mie_field::Mie_field(int lambda, int r, double step, double eps_1, int size, int e0)
{
    double R = r * step;
    double K = 2 * PI / lambda;
    double N1 = std::sqrt(eps_1);
    double N2 = 1;
    int E0 = e0;
    double STEP = step;
    int SIZE = size;
    double XYZ[SIZE];
    for(int i=1; i<=size; i++){
        XYZ[i-1]= 200*i*step - 200*step;
    }
}


Mie_field::Mie_field()
{   
    double STEP = 10e-6;
    int SIZE = 200;
    double R = 50 * STEP;
    double K = 2 * PI / 400;
    double N1 = std::sqrt(0.04);
    double N2 = 1.0;
    int E0 = 100;
    double XYZ[200];
    for(int i=1; i<=SIZE; i++){
        XYZ[i-1]= 200*i*STEP - 200*STEP;
    }
}

int q(int a, double b){
    return a*b;
}



int main(){
    // Mie_field cls; //(int 400, int 50, double 10e-6, double 0.04, int 200, int 100);
    // cls.print();

    std::cout << q(10, 0.2) << std::endl;

    return 0;
}
