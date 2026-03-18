#include "ManyBody.h"
#include "Hamiltonian.h"
#include <Eigen/Core>
#include <iostream>

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