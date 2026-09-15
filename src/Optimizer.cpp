#include "Optimizer.h"

#include <cmath>
#include <stdexcept>

using namespace std;

Optimizer::Optimizer(OptimizerType type, double learningRate, double beta1, double beta2, double epsilon): 
type(type), learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), timestep(0)
{
    if (learningRate <= 0.0)
    {
        throw invalid_argument("Learning rate must be greater than zero");
    }

    if (beta1 < 0.0 || beta1 >= 1.0)
    {
        throw invalid_argument("beta1 must be between 0 and 1");
    }

    if (beta2 < 0.0 || beta2 >= 1.0)
    {
        throw invalid_argument("beta2 must be between 0 and 1");
    }

    if (epsilon <= 0.0)
    {
        throw invalid_argument("Epsilon must be greater than zero");
    }
}

void Optimizer::startStep()
{
    /*
        One call to startStep() represents ONE training step.

        For example:

            epoch
                |
                +-- update layer 0
                |
                +-- update layer 1

        Both layers use the same timestep.
    */

    if (type == OptimizerType::Momentum || type == OptimizerType::Adam)
    {
        timestep++;
    }
}


void Optimizer::update( size_t layerIndex, Matrix& W, Matrix& b, const Matrix& dW, const Matrix& db)
{
    if (W.rows() != dW.rows() || W.cols() != dW.cols())
    {
        throw invalid_argument("Weight and weight-gradient dimensions must match");
    }

    if (b.rows() != db.rows() || b.cols() != db.cols())
    {
        throw invalid_argument("Bias and bias-gradient dimensions must match");
    }


    // Make sure the state vectors contain this layer
    while (velocityW.size() <= layerIndex)
    {
        velocityW.emplace_back();
        velocityB.emplace_back();

        momentW.emplace_back();
        momentB.emplace_back();

        secondMomentW.emplace_back();
        secondMomentB.emplace_back();
    }

    // SGD update
    if (type == OptimizerType::SGD)
    {
        W = W - dW * learningRate;
        b = b - db * learningRate;

        return;
    }

    // Initialize Momentum / Adam states
    if (type == OptimizerType::Momentum)
    {
        if (velocityW[layerIndex].rows() == 0)
        {
            velocityW[layerIndex] = Matrix::zeros(W.rows(), W.cols());
            velocityB[layerIndex] = Matrix::zeros(b.rows(), b.cols());
        }
    }

    if (type == OptimizerType::Adam)
    {
        if (momentW[layerIndex].rows() == 0)
        {
            momentW[layerIndex] = Matrix::zeros(W.rows(), W.cols());
            momentB[layerIndex] = Matrix::zeros(b.rows(), b.cols());

            secondMomentW[layerIndex] = Matrix::zeros(W.rows(), W.cols());
            secondMomentB[layerIndex] = Matrix::zeros(b.rows(), b.cols());
        }
    }
    
    // Momentum update
    if (type == OptimizerType::Momentum)
    {
        /*
            v = beta1 * v + (1 - beta1) * gradient

            parameters = parameters - learningRate * v
        */

        velocityW[layerIndex] = velocityW[layerIndex] * beta1 + dW * (1.0 - beta1);
        velocityB[layerIndex] = velocityB[layerIndex] * beta1 + db * (1.0 - beta1);

        W = W - velocityW[layerIndex] * learningRate;
        b = b - velocityB[layerIndex] * learningRate;

        return;
    }

    // Adam update
    if (type == OptimizerType::Adam)
    {

        /*
            First moment: m = beta1 * m + (1 - beta1) * gradient
            Second moment: v = beta2 * v + (1 - beta2) * gradient^2

            Bias correction: 
            m_hat = m / (1 - beta1^t) 
            v_hat = v / (1 - beta2^t)
        */

        momentW[layerIndex] = momentW[layerIndex] * beta1 + dW * (1.0 - beta1);
        momentB[layerIndex] = momentB[layerIndex] * beta1 + db * (1.0 - beta1);

        // Gradient squared for second moment
        Matrix dWSquared = dW.apply([](double x){return x * x;});
        Matrix dbSquared =db.apply([](double x){return x * x;});


        secondMomentW[layerIndex] = secondMomentW[layerIndex] * beta2 + dWSquared * (1.0 - beta2);
        secondMomentB[layerIndex] = secondMomentB[layerIndex] * beta2 + dbSquared * (1.0 - beta2);

        // Bias correction
        double biasCorrection1 = 1.0 - pow(beta1, static_cast<double>(timestep));
        double biasCorrection2 = 1.0 - pow(beta2, static_cast<double>(timestep));
        

        // Corrected first moments
        Matrix correctedMomentW = momentW[layerIndex] * (1.0 / biasCorrection1);
        Matrix correctedMomentB = momentB[layerIndex] * (1.0 / biasCorrection1);


        // Corrected second moments
        Matrix correctedSecondMomentW = secondMomentW[layerIndex] * (1.0 / biasCorrection2);
        Matrix correctedSecondMomentB = secondMomentB[layerIndex] * (1.0 / biasCorrection2);


        Matrix inverseDenominatorW = correctedSecondMomentW.apply([this](double x){return 1.0 / (sqrt(x) + epsilon);});
        Matrix inverseDenominatorB = correctedSecondMomentB.apply([this](double x){return 1.0 / (sqrt(x) + epsilon);});


        // Update parameters
        Matrix weightUpdate = correctedMomentW.hadamard(inverseDenominatorW);
        Matrix biasUpdate = correctedMomentB.hadamard(inverseDenominatorB);


        W = W - weightUpdate * learningRate;

        b = b - biasUpdate * learningRate;

        return;
    }

    throw invalid_argument("Unsupported optimizer type");
}


void Optimizer::reset()
{
    timestep = 0;

    velocityW.clear();
    velocityB.clear();

    momentW.clear();
    momentB.clear();

    secondMomentW.clear();
    secondMomentB.clear();
}

void Optimizer::setLearningRate(double learningRate)
{
    if (learningRate <= 0.0)
    {
        throw invalid_argument("Learning rate must be greater than zero");
    }

    this->learningRate = learningRate;
}

double Optimizer::getLearningRate() const
{
    return learningRate;
}

size_t Optimizer::getTimestep() const
{
    return timestep;
}