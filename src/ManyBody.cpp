#include "ManyBody.h"

#include "ADC.h"
#include "Hamiltonian.h"
#include "utility.h"

#include <Eigen/Core>
#include <iostream>
#include <Eigen/Dense>

spin_orbital::Amplitudes spin_orbital::SDaction(const spin_orbital::Hamiltonian& hamiltonian,
                                                const spin_orbital::Amplitudes& amplitudes, bool oneElectron,
                                                bool twoElectron) {
  spin_orbital::Amplitudes result(amplitudes);
  const auto& t2 = amplitudes.t2;
  const auto& no = hamiltonian.nelec;
  const auto& nv = hamiltonian.norb - no;
  result.t1.setZero();
  result.t2.setZero();
  for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j)
      for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b) {
          if (oneElectron) {
            for (int m = 0; m < no; ++m)
              result.t2(a, b, i, j) += hamiltonian.f(m, j) * t2(a, b, m, i) - hamiltonian.f(m, i) * t2(a, b, m, j);
            for (int e = 0; e < nv; ++e)
              result.t2(a, b, i, j) +=
                  hamiltonian.f(a + no, e + no) * t2(e, b, i, j) - hamiltonian.f(b + no, e + no) * t2(e, a, i, j);
          }
          if (twoElectron) {
            for (int m = 0; m < no; ++m)
              for (int n = 0; n < m; ++n)
                result.t2(a, b, i, j) += hamiltonian.dirac(m, n, i, j) * t2(a, b, m, n);
            for (int e = 0; e < nv; ++e)
              for (int f = 0; f < e; ++f)
                result.t2(a, b, i, j) += hamiltonian.dirac(e + no, f + no, a + no, b + no) * t2(e, f, i, j);
            for (int e = 0; e < nv; ++e)
              for (int m = 0; m < no; ++m)
                result.t2(a, b, i, j) += hamiltonian.dirac(a + no, m, i, e + no) * t2(e, b, m, j) -
                    hamiltonian.dirac(b + no, m, i, e + no) * t2(e, a, m, j) -
                    hamiltonian.dirac(a + no, m, j, e + no) * t2(e, b, m, i) +
                    hamiltonian.dirac(b + no, m, j, e + no) * t2(e, a, m, i);
          }
        }
  return result;
}

Eigen::VectorXd spin_orbital::PAPT_action(const spin_orbital::Amplitudes& amplitudes,
                                          const spin_orbital::Amplitudes& action) {
  const auto& no = amplitudes.t1.dimension(1);
  const auto& nv = amplitudes.t1.dimension(0);
  spin_orbital::Hamiltonian result(no + nv, true, false);
  result.nelec = no;
  result.spin_orbital_symmetries = amplitudes.reference_hamiltonian.spin_orbital_symmetries;
  //  std::cout << "amplitudes "<<amplitudes.t2<<std::endl;
  //  std::cout << "action "<<action.t2<<std::endl;
  result.h.setZero();
  for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j)
      for (int m = 0; m < no; ++m)
        for (int e = 0; e < nv; ++e)
          for (int f = 0; f < nv; ++f)
            result.h(i, j) += double(0.5) * amplitudes.t2(e, f, i, m) * action.t2(e, f, m, j) +
                double(0.5) * amplitudes.t2(e, f, j, m) * action.t2(e, f, m, i);
  for (int a = 0; a < nv; ++a)
    for (int b = 0; b < nv; ++b)
      for (int c = 0; c < nv; ++c)
        for (int m = 0; m < no; ++m)
          for (int n = 0; n < no; ++n)
            result.h(no + a, no + b) += double(0.5) * amplitudes.t2(a, c, m, n) * action.t2(c, b, m, n) +
                double(0.5) * amplitudes.t2(b, c, m, n) * action.t2(c, a, m, n);
  return PAPT_pack(result);
}

Eigen::VectorXd spin_orbital::PAPT_pack(const spin_orbital::Hamiltonian& hamiltonian) {
  const auto& no = hamiltonian.nelec;
  const auto& nv = hamiltonian.norb - hamiltonian.nelec;
  int npack = 0;
  for (int i = 0; i < no; ++i)
    for (int j = 0; j <= i; ++j)
      if (hamiltonian.spin_orbital_symmetries[i] == hamiltonian.spin_orbital_symmetries[j])
        ++npack;
  for (int a = 0; a < nv; ++a)
    for (int b = 0; b <= a; ++b)
      if (hamiltonian.spin_orbital_symmetries[a + no] == hamiltonian.spin_orbital_symmetries[b + no])
        ++npack;
  Eigen::VectorXd res(npack);
  size_t off = 0;
  for (int i = 0; i < no; ++i)
    for (int j = 0; j <= i; ++j)
      if (hamiltonian.spin_orbital_symmetries[i] == hamiltonian.spin_orbital_symmetries[j])
        res[off++] = hamiltonian.h(i, j);
  for (int a = 0; a < nv; ++a)
    for (int b = 0; b <= a; ++b)
      if (hamiltonian.spin_orbital_symmetries[a + no] == hamiltonian.spin_orbital_symmetries[b + no])
        res[off++] = hamiltonian.h(no + a, no + b);
  return res;
}

spin_orbital::Hamiltonian spin_orbital::PAPT_unpack(const Eigen::VectorXd& vector, size_t norb, size_t nelec,
                                                    std::vector<int> spin_orbital_symmetries) {
  spin_orbital::Hamiltonian result(norb, true, false);
  result.h.setZero();
  int no = nelec;
  int nv = norb - nelec;
  size_t off = 0;
  for (int i = 0; i < no; ++i)
    for (int j = 0; j <= i; ++j)
      if (spin_orbital_symmetries[i] == spin_orbital_symmetries[j])
        result.h(i, j) = result.h(j, i) = vector[off++];
  for (int a = 0; a < nv; ++a)
    for (int b = 0; b <= a; ++b)
      if (spin_orbital_symmetries[a + no] == spin_orbital_symmetries[b + no])
        result.h(no + a, no + b) = result.h(no + b, no + a) = vector[off++];
  result.f = result.h;
  result.nelec = nelec;
  result.e0 = 0;
  for (int i = 0; i < nelec; ++i)
    result.e0 += result.f(i, i);
  return result;
}

spin_orbital::Hamiltonian spin_orbital::PAPT_unpack(const Eigen::VectorXd& vector,
                                                    const spin_orbital::Hamiltonian& reference_operator) {
  auto hamiltonian =
      spin_orbital::PAPT_unpack(vector, reference_operator.norb, reference_operator.nelec,
                                reference_operator.spin_orbital_symmetries);
  hamiltonian.spin_orbital = reference_operator.spin_orbital;
  hamiltonian.ecore = reference_operator.ecore;
  hamiltonian.uhf = reference_operator.uhf;
  hamiltonian.occ = reference_operator.occ;
  hamiltonian.closed = reference_operator.closed;
  hamiltonian.orbsym = reference_operator.orbsym;
  hamiltonian.spin_multiplicity = reference_operator.spin_multiplicity;
  return hamiltonian;
}

Eigen::VectorXd spin_orbital::PAPT_kernel_action(const Eigen::VectorXd& potential,
                                                 const spin_orbital::Amplitudes& amplitudes) {
  auto hamiltonian = PAPT_unpack(potential, amplitudes.reference_hamiltonian);
  auto action = SDaction(hamiltonian, amplitudes, true, false);
  return PAPT_action(amplitudes, action);
}

spin_orbital::Hamiltonian dress_hamiltonian(const spin_orbital::Hamiltonian& hamiltonian) {
  spin_orbital::Hamiltonian result = hamiltonian;
  auto no = hamiltonian.nelec;
  auto nv = hamiltonian.norb - no;
  spin_orbital::Amplitudes Kijab(hamiltonian);
  auto amplitudes = Kijab.MP1(hamiltonian);
  size_t off = 0;
  for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j) {
      for (int k = 0; k < no; ++k)
        for (int a = 0; a < nv; ++a)
          for (int b = 0; b <= a; ++b)
            result.f(i, j) += 0.5 * amplitudes.t2(a, b, i, k) * Kijab.t2(a, b, k, j);
      for (int a = 0; a < nv; ++a)
        for (int m = 0; m < no; ++m)
          for (int n = 0; m < no; ++n)
            result.f(i, j) -= 0.5
                * hamiltonian.dirac(i, a + no, m, n)
                * hamiltonian.dirac(j, a + no, m, n)
                / (
                  hamiltonian.f(a + nv, a + nv)
                  + hamiltonian.f(i, i)
                  - hamiltonian.f(m, m)
                  - hamiltonian.f(n, m));
    }
  for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j)
      result.f(i, j) = 0.5 * (result.f(i, j) + result.f(j, i));

  for (int a = 0; a < nv; ++a)
    for (int b = 0; b <= a; ++b)
      for (int c = 0; c < nv; ++c)
        for (int i = 0; i < no; ++i) {
          for (int j = 0; j < no; ++j)
            result.f(a + no, b + no) += 0.5 * amplitudes.t2(a, c, i, j) * Kijab.t2(c, b, i, j);
          for (int d = 0; d < nv; ++d)
            result.f(a + no, b + no) -= 0.5 * hamiltonian.dirac(c + no, d + no, a + no, i) * hamiltonian.dirac(
                    c + no, d + no, b + no, i)
                / (hamiltonian.f(c + no, c + no) + hamiltonian.f(d + no, d + no) - hamiltonian.f(a + no, a + no) -
                   hamiltonian.f(i, i));
        }
  for (int a = 0; a < nv; ++a)
    for (int b = 0; b <= a; ++b)
      result.f(no + a, no + b) = 0.5 * (result.f(no + a, no + b) + result.f(no + b, no + a));
  return result;
}

void spin_orbital::calculate_results(const std::string& method, const spin_orbital::Hamiltonian& modified_hamiltonian,
                                     molpro::PluginGuest& molproPlugin, const spin_orbital::Hamiltonian& hamiltonian,
                                     const
                                     spin_orbital::Amplitudes& Kijab, const spin_orbital::Amplitudes& amplitudes,
                                     double reference_energy_2, double reference_energy_3, const std::string& dump_file,
                                     bool ip,
                                     bool ea) {
  auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(
      Eigen::Map<Eigen::MatrixXd>(const_cast<double*>(modified_hamiltonian.f.data()), modified_hamiltonian.norb,
                                  modified_hamiltonian.norb));
  //    std::cout << method+" operator eigenvalues: " << solver.eigenvalues().transpose() << std::endl;
  auto operator_rotated = modified_hamiltonian;
  Eigen::VectorXd eigval = solver.eigenvalues().eval();
  auto eigenvectors = solver.eigenvectors().eval();
  utility::eigensolution_sort(eigenvectors, eigval);
  std::cout << method + " operator eigenvalues: " << eigval.transpose() << std::endl;
  for (int i = modified_hamiltonian.nelec; i < hamiltonian.norb; ++i)
    if (eigval(i) < eigval(modified_hamiltonian.nelec - 1))
      eigval(i) = 1e99; // project out intruders
  const Eigen::MatrixXd eigvalmat = eigval.asDiagonal();
  operator_rotated.f =
      Eigen::TensorMap<const Eigen::Tensor<double, 2>>(eigvalmat.data(), eigval.rows(), eigval.rows());
  auto rotated_Kijab = Kijab.transform(eigenvectors);
  auto amplitudes_PAPT1 = rotated_Kijab.MP1(operator_rotated);
  double epapt2 = rotated_Kijab * amplitudes_PAPT1;
  std::cout << method + "2 energy contribution and total: " << epapt2 << " "
      << hamiltonian.e0 + hamiltonian.e1 + epapt2
      << std::endl;
  auto amplitudes_PAPT1_backrotated = amplitudes_PAPT1.transform(eigenvectors.transpose());
  std::cout << method + "2 energy contribution: "
      << -(SDaction(modified_hamiltonian, amplitudes_PAPT1_backrotated, true, false) *
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
    modified_hamiltonian.dump(dump_file);

  spin_orbital::result(molproPlugin, method + "2", hamiltonian.e0 + hamiltonian.e1 + epapt2);
  spin_orbital::result(molproPlugin, method + "3", hamiltonian.e0 + hamiltonian.e1 + epapt2 + epapt3);
  //        result(molproPlugin, "MP2", hamiltonian.e0 + hamiltonian.e1 + emp2);
  //        result(molproPlugin, "MP3", hamiltonian.e0 + hamiltonian.e1 + emp2 + emp3);

  if (ip) {
    spin_orbital::result(molproPlugin, "IP-ADC(0)", IP_ADC(hamiltonian, amplitudes, 0));
    spin_orbital::result(molproPlugin, "IP-ADC(2)", IP_ADC(hamiltonian, amplitudes, 2));
    spin_orbital::result(molproPlugin, "IP-ADC(P2)", IP_ADC(hamiltonian, amplitudes_PAPT1_backrotated, 2));
  }
  if (ea) {
    spin_orbital::result(molproPlugin, "EA-ADC(0)", EA_ADC(hamiltonian, amplitudes, 0));
    spin_orbital::result(molproPlugin, "EA-ADC(2)", EA_ADC(hamiltonian, amplitudes, 2));
    spin_orbital::result(molproPlugin, "EA-ADC(P2)", EA_ADC(hamiltonian, amplitudes_PAPT1_backrotated, 2));
  }
}