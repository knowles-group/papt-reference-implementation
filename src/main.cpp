#include "Amplitudes.h"
#include "Hamiltonian.h"
#include "ManyBody.h"
#include <molpro/FCIdump.h>
#include <molpro/PluginGuest.h>
#include "mpi.h"
#include "utility.h"
#include <iomanip>
#include <iostream>
#include "PAPT.h"
#include "dress.h"

using molpro::FCIdump;
using namespace spin_orbital;


Hamiltonian setup_hamiltonian(int argc, char** argv, molpro::PluginGuest molproPlugin) {
  std::string fcidumpname;
  if (molproPlugin.active()) {
    molproPlugin.send("GIVE OPERATOR HAMILTONIAN FCIDUMP");
    fcidumpname = molproPlugin.receive();
  } else {
    if (argc < 2)
      throw std::out_of_range("must give FCIdump filename as first command-line argument");
    fcidumpname = argv[1];
  }
  std::cout << "fcidump " << fcidumpname << std::endl;
  auto fcidump = molpro::FCIdump(fcidumpname);
  return Hamiltonian(fcidump);
}


int main(int argc, char* argv[]) {
  std::cout << std::fixed << std::setprecision(8);
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  molpro::PluginGuest molproPlugin;
  Hamiltonian hamiltonian = setup_hamiltonian(argc, argv, molproPlugin);

  std::cout << "Reference energy: " << hamiltonian.e0 + hamiltonian.e1 << std::endl;
  std::cout << "MP0 energy: " << hamiltonian.e0 << std::endl;

  Amplitudes Kijab(hamiltonian);
  auto amplitudes = Kijab.MP1(hamiltonian);
  const auto MP1action1 = SDaction(hamiltonian, amplitudes, true, false);
  const auto MP1action2 = SDaction(hamiltonian, amplitudes, false, true);
  const auto MP1action12 = SDaction(hamiltonian, amplitudes, true, true);

  double emp2 = amplitudes.energy(hamiltonian);
  std::cout << "MP2 energy contribution: " << emp2 << " or " << -(MP1action1 * amplitudes) << std::endl;
  std::cout << "MP2 energy: " << hamiltonian.e0 + hamiltonian.e1 + emp2 << std::endl;

  double emp3 = amplitudes * MP1action2;
  std::cout << "MP3 energy contribution: " << emp3 << std::endl;
  std::cout << "MP3 energy: " << hamiltonian.e0 + hamiltonian.e1 + emp2 + emp3 << std::endl;

  if (false)
    compute_papt_results(molproPlugin, hamiltonian, Kijab, amplitudes, MP1action12, argc > 3 ? std::stod(argv[3]) : 0,
                         argc > 4 ? std::stod(argv[4]) : 0,
                         argc > 2 ? argv[2] : "");
  compute_dress_results(molproPlugin, hamiltonian, Kijab, amplitudes, MP1action12, argc > 3 ? std::stod(argv[3]) : 0,
                        argc > 4 ? std::stod(argv[4]) : 0,
                        argc > 2 ? argv[2] : "");

  MPI_Finalize();
  return 0;
}