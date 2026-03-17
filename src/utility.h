#ifndef SPIN_ORBITAL__LA_H_
#define SPIN_ORBITAL__LA_H_
#include <Eigen/Core>
#include <molpro/PluginGuest.h>

namespace spin_orbital
{
    Eigen::VectorXd linsolve(const Eigen::MatrixXd& kernel, const Eigen::VectorXd& rhs);
    void result(const molpro::PluginGuest& molproPlugin, const std::string& name, double value);
}

namespace utility
{
    void eigensolution_sort(Eigen::MatrixXd& vec, Eigen::VectorXd& val);
}
#endif // SPIN_ORBITAL__LA_H_
