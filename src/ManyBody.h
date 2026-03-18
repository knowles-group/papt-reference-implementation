#ifndef SPIN_ORBITAL__MANYBODY_H_
#define SPIN_ORBITAL__MANYBODY_H_
#include "Amplitudes.h"
namespace spin_orbital {
Amplitudes SDaction(const spin_orbital::Hamiltonian& hamiltonian, const spin_orbital::Amplitudes& amplitudes,
                    bool oneElectron = true, bool twoElectron = true);

Eigen::VectorXd PAPT_action(const spin_orbital::Amplitudes& amplitudes, const spin_orbital::Amplitudes& action);

Eigen::VectorXd PAPT_kernel_action(const Eigen::VectorXd& potential, const spin_orbital::Amplitudes& amplitudes);

Eigen::VectorXd PAPT_pack(const spin_orbital::Hamiltonian& hamiltonian);

spin_orbital::Hamiltonian PAPT_unpack(const Eigen::VectorXd& vector, size_t norb, size_t nelec,
                                      std::vector<int> spin_orbital_symmetries);
spin_orbital::Hamiltonian PAPT_unpack(const Eigen::VectorXd& vector,
                                      const spin_orbital::Hamiltonian& reference_operator);

spin_orbital::Hamiltonian dress_hamiltonian(const spin_orbital::Hamiltonian& hamiltonian);

} // namespace spin_orbital
#endif // SPIN_ORBITAL__MANYBODY_H_
