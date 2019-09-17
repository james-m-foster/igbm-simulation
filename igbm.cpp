#include <fstream>
#include <iomanip>
#include <random>
#include <chrono>

using namespace std;

// Numerical simulation of Inhomogeneous Geometric Brownian Motion (IGBM) on [0, T]:
// dy_t = a(b-y_t) dt + sigma y_t dW_t

// The below class contains the five numerical methods considered in the paper:
//
// J. Foster, T. Lyons and H. Oberhauser, An optimal polynomial approximation
// of Brownian motion, arxiv.org/abs/1904.06998, 2019.
class IGBMmethods {
    public:
        // Input parameters
        double a, b, sigma, T;
        int no_of_steps;

        IGBMmethods(double, double, double, double, int);

        double euler_maruyama(double, double);
        double milstein(double, double);
        double linear_ode(double, double);
        double parabola_ode(double, double, double);
        double log_ode(double, double, double);

    private:
        // Weights and nodes used for 3-point Gauss-Legendre quadrature
        const double x1 = 0.5 - sqrt(0.15);
        const double x3 = 0.5 + sqrt(0.15);
        const double x1_squared = pow(x1, 2);
        const double x3_squared = pow(x3, 2);
        const double small_weight = 5.0/18.0;
        const double big_weight = 4.0/9.0;

        // Precomputed values that depend on the input parameters
        double step_size;
        double sigma_squared, half_sigma_squared, six_sigma;
        double drift_const, drift_linear;

        // Precomputed values that are method specific
        double euler_const, milstein_const;
        double log_ode_const_1, log_ode_const_2, log_ode_const_3;
};

// Constructor will compute the above private variables
IGBMmethods::IGBMmethods(double input_a, double input_b, double input_sigma,
                         double input_T, int input_no_of_steps){

    a = input_a;
    b = input_b;
    sigma = input_sigma;
    T = input_T;
    no_of_steps = input_no_of_steps;

    step_size =  input_T/(double)input_no_of_steps;

    sigma_squared = pow(input_sigma, 2);
    half_sigma_squared = 0.5*sigma_squared;
    six_sigma = 6.0*input_sigma;

    drift_const = input_a*input_b*step_size;
    drift_linear = -(input_a + 0.5*sigma_squared)*step_size;

    euler_const = 1.0 - input_a*step_size;
    milstein_const =  euler_const - half_sigma_squared*step_size;

    log_ode_const_1 = drift_const*(1.0 + sigma_squared*(step_size/30.0));
    log_ode_const_2 = -drift_const*input_sigma;
    log_ode_const_3 = 0.6*sigma_squared*drift_const;
};

// Computes one step of the (non-negative) Euler-Maruyama method
double IGBMmethods::euler_maruyama(double y0, double brownian_increment){

    return max(0.0, drift_const + y0*(euler_const + sigma*brownian_increment));
};

// Computes one step of the (non-negative) Milstein method
double IGBMmethods::milstein(double y0, double brownian_increment){

    return max(0.0, drift_const \
                        + y0*(milstein_const + brownian_increment \
                                *(sigma + half_sigma_squared*brownian_increment)));
};

// Computes one step of the standard piecewise linear SDE approximation
double IGBMmethods::linear_ode(double y0, double brownian_increment){

    double drift = drift_linear + sigma*brownian_increment;

    if (drift == 0.0){
        return y0 + drift_const;
    }

    double exp_drift = exp(drift);

    return y0*exp_drift + drift_const*((exp_drift - 1.0)/drift);
};

// Computes one step of the parabola-ODE method
// approximated by 3-point Gauss-Legendre quadrature
double IGBMmethods::parabola_ode(double y0, double brownian_increment,
                                 double brownian_area){

    double drift = drift_linear + sigma*brownian_increment;
    double exp_drift = exp(drift);

    double quadratic_coef = six_sigma*brownian_area;
    double linear_coef = -drift - quadratic_coef;

    // Approximates the "parabola" integral
    // by 3-point Gauss-Legendre quadrature
    double integral = small_weight*exp(x1*linear_coef + x1_squared*quadratic_coef) \
                       + big_weight*exp(0.5*linear_coef + 0.25*quadratic_coef) \
                       + small_weight*exp(x3*linear_coef + x3_squared*quadratic_coef);

    return exp_drift*(y0 + drift_const*integral);
};

// Computes one step of the high order log-ODE method
double IGBMmethods::log_ode(double y0, double brownian_increment, double brownian_area){

    double drift = drift_linear + sigma*brownian_increment;

    if (drift == 0.0){
        return y0 + log_ode_const_1 \
                  + brownian_area*(log_ode_const_2 + log_ode_const_3*brownian_area);
    }

    double exp_drift = exp(drift);

    return y0*exp_drift + (log_ode_const_1 + brownian_area \
                             *(log_ode_const_2 + log_ode_const_3*brownian_area)) \
                                *((exp_drift - 1.0)/drift);
};

int main()
{
    // Input parameters
    const double a = 0.1;
    const double b = 0.04;
    const double sigma = 0.6;
    const double y0 = 0.06;
    const double T = 5.0;
    const int no_of_steps = 100;

    // Number of steps used by the "fine" approximation
    // during each step of the "crude" numerical method
    const int no_of_fine_steps = 10;

    // Number of paths used for the Monte Carlo estimators
    const int no_of_paths = 100000;

    // Variance for generating the "time area" normal random variables
    const double twelve = 1.0/12.0;

    // Step size parameters
    const double step_size =  T/(double)no_of_steps;
    const double one_over_step_size = 1.0/step_size;
    const double fine_step_size = T/(double)(no_of_steps*no_of_fine_steps);

    // We will be comparing the methods on two different time scales
    IGBMmethods course_method(a, b, sigma, T, no_of_steps);
    IGBMmethods fine_method(a, b, sigma, T, no_of_steps*no_of_fine_steps);

    // Normal distributions for generating the various increments and time areas
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> increment_distribution(0.0, sqrt(step_size));
    std::normal_distribution<double> area_distribution(0.0, sqrt(twelve*step_size));
    std::normal_distribution<double> fine_increment_distribution(0.0, sqrt(fine_step_size));
    std::normal_distribution<double> fine_area_distribution(0.0, sqrt(twelve \
                                                                      *fine_step_size));

    // Numerical solutions computed with course and fine step sizes
    double y_1 = y0;
    double y_2 = y0;
    double y_fine = y0;

    // Information about the Brownian motion (increments and areas)
    // These objects are described in polynomial_presentation.pdf as:
    // brownian_increment is W_{s,t}
    // brownian_area      is H_{s,t}
    double brownian_increment = 0.0;
    double brownian_area = 0.0;
    double fine_brownian_increment = 0.0;
    double fine_brownian_area = 0.0;

    // Strong error estimators for y at time T
    double end_point_error_1 = 0.0;
    double end_point_error_2 = 0.0;

    // Weak error estimators for y at time T
    double call_option_error_1 = 0.0;
    double call_option_error_2 = 0.0;
    double call_option_price = 0.0;

    double samplepath[no_of_steps + 1] = {y0};

    for (int i=0; i<no_of_paths; ++i) {
        for (int j=1; j<=no_of_steps; ++j) {

            brownian_increment = 0.0;
            brownian_area = 0.0;

            for (int k=1; k<= no_of_fine_steps; ++k){
                // Generate information about the Brownian path over the "fine" increment
                fine_brownian_increment = fine_increment_distribution(generator);
                fine_brownian_area = fine_area_distribution(generator);

                // Propagate the numerical solution over the fine increment
                y_fine = fine_method.log_ode(y_fine, fine_brownian_increment,
                                                     fine_brownian_area);

                // Update the information about the Brownian path over the
                // course increment using the recently generated variables.
                // The below procedure can be derived using some elementary
                // properties of integration (additivity and linearity)
                brownian_area = brownian_area + fine_step_size
                                 * (brownian_increment + 0.5*fine_brownian_increment \
                                                       + fine_brownian_area);

                brownian_increment = brownian_increment + fine_brownian_increment;
            }

            // Compute the time area for the Brownian path over the course increment
            brownian_area = brownian_area*one_over_step_size - 0.5*brownian_increment;

            // Propagate the numerical solutions over the course increment
            y_1 = course_method.log_ode(y_1, brownian_increment, brownian_area);
            y_2 = course_method.parabola_ode(y_2, brownian_increment, brownian_area);

            // Store the sample path if we have reached the final iteration
            if (i == no_of_paths - 1){
                samplepath[j] = y_1;
            }
        }

        // Compute the L2 error between the methods on the fine and course scales
        end_point_error_1 = end_point_error_1 + pow(y_1 - y_fine, 2);
        end_point_error_2 = end_point_error_2 + pow(y_2 - y_fine, 2);

        // Evaluate the call option payoffs
        call_option_error_1 = call_option_error_1 + max(0.0, y_1 - b);
        call_option_error_2 = call_option_error_2 + max(0.0, y_2 - b);
        call_option_price = call_option_price + max(0.0, y_fine - b);

        // Reset the numerical solutions
        y_1 = y0;
        y_2 = y0;
        y_fine = y0;
    }

    // Compute the various averages for estimating the strong and weak errors
    end_point_error_1 = sqrt(end_point_error_1 / (double(no_of_paths)));
    end_point_error_2 = sqrt(end_point_error_2 / (double(no_of_paths)));

    call_option_error_1 = call_option_error_1 / (double(no_of_paths));
    call_option_error_2 = call_option_error_2 / (double(no_of_paths));
    call_option_price = call_option_price / (double(no_of_paths));

    // Initialize the numerical solution for the speed test
    double y_test = y0;

    // Start the speed test
    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<no_of_paths; ++i) {
        for (int j=1; j<=no_of_steps; ++j) {

            // Generate information about Brownian path
            brownian_increment = increment_distribution(generator);
            brownian_area = area_distribution(generator);

            // Propagate the numerical solution over the course increment
            y_test = course_method.log_ode(y_test, brownian_increment, brownian_area);
        }

        // Reset the numerical solution
        y_test = y0;
    }

    // End the speed test
    auto finish = std::chrono::high_resolution_clock::now();

    // Obtain the time taken by the speed test
    std::chrono::duration<double> elapsed = finish - start;

    // Display the results in a text file
    ofstream myfile;
    myfile.open ("igbm_simulation.txt");

    myfile << std::fixed << std::setprecision(2) << "L2 error at time T = " << T \
           << " for method 1: \t " << std::setprecision(15) << end_point_error_1
           << "\t" << "L2 error at time T = " << std::setprecision(2) << T \
           << " for method 2: \t " << std::setprecision(15) << end_point_error_2 << "\n";

    myfile << std::fixed << std::setprecision(2) \
           << "Call option error at time T = " << T << " for method 1: " \
           << std::setprecision(15) << abs(call_option_error_1 - call_option_price) \
           << "\t" << "Call option error at time T = " << std::setprecision(2) \
           << T << " for method 2: " << std::setprecision(15) \
           << abs(call_option_error_2 - call_option_price) << "\n";

    myfile << std::fixed << std::setprecision(2) \
           << "Call option value computed at time T = " << T << ": \t " \
           << std::setprecision(15) << call_option_price << "\n\n";

    myfile << std::fixed << std::setprecision(15) \
           << "Number of steps: " << "\t\t" << no_of_steps << "\n";

    myfile << std::fixed << std::setprecision(10) \
           << "Number of sample paths: " << "\t" << no_of_paths << "\n";

    myfile << std::fixed << std::setprecision(10) \
           << "Time taken in speed test: " << "\t" << elapsed.count() << "\n\n";

    myfile << std::fixed << std::setprecision(15) \
           << "Example sample path for first method" << "\n\n" ;

    myfile << std::fixed << std::setprecision(15) \
            << "t" << "\t\t\t" << "y_t" << "\n";

    for (int j=0; j<=no_of_steps; ++j) {
        myfile << std::fixed << std::setprecision(15) \
               << j*step_size << "\t" << samplepath[j] << "\n";
    }

    myfile.close();

    return 0;
}
