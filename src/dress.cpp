#include "dress.h"
#include "ManyBody.h"
#include "utility.h"
#include <iostream>
#include <Eigen/Dense>

namespace spin_orbital {


void compute_dress_results(molpro::PluginGuest& molproPlugin, Hamiltonian& hamiltonian,
                          const Amplitudes& Kijab, const Amplitudes& amplitudes, const Amplitudes MP1action12,
                          double reference_energy_2, double
                          reference_energy_3, const std::string& dump_file) {
  bool ip = false;
  bool ea = false;

  auto papt_operator = spin_orbital::dress_hamiltonian(hamiltonian);
if (false) {

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
}
  calculate_results("dressed", papt_operator, molproPlugin, hamiltonian, Kijab, amplitudes, reference_energy_2,
                    reference_energy_3, dump_file, ip, ea);
}
}