#ifndef REFERENCE_IMPLEMENTATION_DRESS_HPP
#define REFERENCE_IMPLEMENTATION_DRESS_HPP
#include "Hamiltonian.h"
#include "Amplitudes.h"
#include <molpro/PluginGuest.h>

namespace spin_orbital {
void compute_dress_results(molpro::PluginGuest& molproPlugin, Hamiltonian& hamiltonian,
                          const Amplitudes& Kijab, const Amplitudes& amplitudes, const Amplitudes MP1action12, double reference_energy_2 = 0, double
                          reference_energy_3 = 0, const std::string& dump_file = {});
}
#endif //REFERENCE_IMPLEMENTATION_DRESS_HPP