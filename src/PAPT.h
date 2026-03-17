#ifndef REFERENCE_IMPLEMENTATION_PAPT_HPP
#define REFERENCE_IMPLEMENTATION_PAPT_HPP
#include "Hamiltonian.h"
#include "Amplitudes.h"
#include <molpro/PluginGuest.h>

namespace spin_orbital {
void compute_papt_results(int argc, char** argv, molpro::PluginGuest molproPlugin, Hamiltonian hamiltonian,
                          Amplitudes Kijab, Amplitudes amplitudes, const Amplitudes MP1action12);
}
#endif //REFERENCE_IMPLEMENTATION_PAPT_HPP