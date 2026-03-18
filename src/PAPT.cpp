#include "PAPT.h"
#include "ADC.h"
#include "ManyBody.h"
#include "utility.h"
#include <iostream>
#include <Eigen/Dense>

namespace spin_orbital {
void calculate_results(const std::string& method, const spin_orbital::Hamiltonian& papt_operator,
                       molpro::PluginGuest& molproPlugin, const Hamiltonian& hamiltonian, const
                       Amplitudes& Kijab, const Amplitudes& amplitudes,
                       double reference_energy_2, double reference_energy_3, const std::string& dump_file, bool ip,
                       bool ea) {
  auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(
      Eigen::Map<Eigen::MatrixXd>(const_cast<double*>(papt_operator.f.data()), papt_operator.norb, papt_operator.norb));
  //    std::cout << method+" operator eigenvalues: " << solver.eigenvalues().transpose() << std::endl;
  auto papt_operator_rotated = papt_operator;
  Eigen::VectorXd eigval = solver.eigenvalues().eval();
  auto eigenvectors = solver.eigenvectors().eval();
  utility::eigensolution_sort(eigenvectors, eigval);
  std::cout << method + " operator eigenvalues: " << eigval.transpose() << std::endl;
  for (int i = papt_operator.nelec; i < hamiltonian.norb; ++i)
    if (eigval(i) < eigval(papt_operator.nelec - 1))
      eigval(i) = 1e99; // project out intruders
  const Eigen::MatrixXd eigvalmat = eigval.asDiagonal();
  papt_operator_rotated.f =
      Eigen::TensorMap<const Eigen::Tensor<double, 2>>(eigvalmat.data(), eigval.rows(), eigval.rows());
  auto rotated_Kijab = Kijab.transform(eigenvectors);
  auto amplitudes_PAPT1 = rotated_Kijab.MP1(papt_operator_rotated);
  double epapt2 = rotated_Kijab * amplitudes_PAPT1;
  std::cout << method + "2 energy contribution and total: " << epapt2 << " "
      << hamiltonian.e0 + hamiltonian.e1 + epapt2
      << std::endl;
  auto amplitudes_PAPT1_backrotated = amplitudes_PAPT1.transform(eigenvectors.transpose());
  std::cout << method + "2 energy contribution: "
      << -(SDaction(papt_operator, amplitudes_PAPT1_backrotated, true, false) *
           amplitudes_PAPT1_backrotated)
      << std::endl;
  double epapt2_alt = Kijab * amplitudes_PAPT1_backrotated;
  std::cout << method + "2 energy contribution and total: " << epapt2 << " "
      << hamiltonian.e0 + hamiltonian.e1 + epapt2_alt
      << std::endl;
  double epapt3 =
      SDaction(hamiltonian, amplitudes_PAPT1_backrotated, true, true) * amplitudes_PAPT1_backrotated + epapt2;
  std::cout << method + "3 energy contribution and total: " << epapt3 << " "
      << hamiltonian.e0 + hamiltonian.e1 + epapt2 + epapt3 << std::endl;
  if ((reference_energy_2 != 0 && std::abs(epapt2 - reference_energy_2) > 1e-6) || std::isnan(epapt2))
    throw std::runtime_error(
        std::string{"Second order energy does not match reference value "} + std::to_string(reference_energy_2));
  if ((reference_energy_3 != 0 && std::abs(epapt3 - reference_energy_3) > 1e-6) || std::isnan(epapt3))
    throw std::runtime_error(
        std::string{"Third order energy does not match reference value "} + std::to_string(reference_energy_3));

  if (!dump_file.empty())
    papt_operator.dump(dump_file);

  result(molproPlugin, method + "2", hamiltonian.e0 + hamiltonian.e1 + epapt2);
  result(molproPlugin, method + "3", hamiltonian.e0 + hamiltonian.e1 + epapt2 + epapt3);
  //        result(molproPlugin, "MP2", hamiltonian.e0 + hamiltonian.e1 + emp2);
  //        result(molproPlugin, "MP3", hamiltonian.e0 + hamiltonian.e1 + emp2 + emp3);

  if (ip) {
    result(molproPlugin, "IP-ADC(0)", IP_ADC(hamiltonian, amplitudes, 0));
    result(molproPlugin, "IP-ADC(2)", IP_ADC(hamiltonian, amplitudes, 2));
    result(molproPlugin, "IP-ADC(P2)", IP_ADC(hamiltonian, amplitudes_PAPT1_backrotated, 2));
  }
  if (ea) {
    result(molproPlugin, "EA-ADC(0)", EA_ADC(hamiltonian, amplitudes, 0));
    result(molproPlugin, "EA-ADC(2)", EA_ADC(hamiltonian, amplitudes, 2));
    result(molproPlugin, "EA-ADC(P2)", EA_ADC(hamiltonian, amplitudes_PAPT1_backrotated, 2));
  }
}

void compute_papt_results(molpro::PluginGuest& molproPlugin, Hamiltonian& hamiltonian,
                          const Amplitudes& Kijab, const Amplitudes& amplitudes, const Amplitudes MP1action12,
                          double reference_energy_2, double
                          reference_energy_3, const std::string& dump_file) {
  bool ip = false;
  bool ea = false;

  const auto papt_rhs = PAPT_action(amplitudes, MP1action12);

  auto papt_dimension = papt_rhs.rows();
  Eigen::MatrixXd papt_kernel(papt_dimension, papt_dimension);
  for (size_t i = 0; i < papt_dimension; ++i) {
    Eigen::VectorXd value(papt_dimension);
    value.setZero();
    value[i] = 1;
    auto line = PAPT_kernel_action(value, amplitudes);
    for (size_t j = 0; j < papt_dimension; ++j)
      papt_kernel(j, i) = line(j);
  }

  //  std::cout << "papt_rhs\n" << papt_rhs << std::endl;
  //  std::cout << "papt_kernel\n" << papt_kernel << std::endl;
  auto papt_solution = linsolve(papt_kernel, papt_rhs);
  //  std::cout << "papt_solution\n" << papt_rhs << std::endl;

  std::cout << "check: " << (papt_kernel * papt_solution - papt_rhs).norm() << std::endl;
  auto papt_operator = PAPT_unpack(papt_solution, hamiltonian);
  auto solver_raw = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(
      Eigen::Map<Eigen::MatrixXd>(papt_operator.f.data(), papt_operator.norb, papt_operator.norb));
  Eigen::VectorXd eigval_raw = solver_raw.eigenvalues().eval();
  auto eigenvectors_raw = solver_raw.eigenvectors().eval();
  utility::eigensolution_sort(eigenvectors_raw, eigval_raw);
  papt_operator.e0 = 0;
  for (int i = 0; i < papt_operator.nelec; ++i)
    papt_operator.e0 += eigval_raw(i);
  //    std::cout << method+" operator eigenvalues before shift: " <<
  //      eigval_raw.transpose() << std::endl; std::cout <<
  //      "hamiltonian.e0 " << hamiltonian.e0 << std::endl; std::cout <<
  //      "papt_operator.e0 " << papt_operator.e0 << std::endl;
  for (int i = 0; i < hamiltonian.norb; ++i)
    papt_operator.f(i, i) += (hamiltonian.e0 - hamiltonian.ecore - papt_operator.e0) / hamiltonian.nelec;
  calculate_results("PAPT", papt_operator, molproPlugin, hamiltonian, Kijab, amplitudes, reference_energy_2,
                    reference_energy_3, dump_file, ip, ea);
}
}